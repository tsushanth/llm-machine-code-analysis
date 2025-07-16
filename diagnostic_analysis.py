#!/usr/bin/env python3
"""
Diagnostic Token Analysis - Debug Version
=========================================
for linux

This version includes detailed debugging to understand what's happening
with the instruction extraction process.
"""

import os
import subprocess
import tempfile
import tiktoken
import re
from typing import Tuple, Optional

class DiagnosticExtractor:
    """Diagnostic version with detailed logging."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        print(f"Working directory: {self.temp_dir}")
    
    def extract_and_debug(self, source_code: str, name: str) -> Tuple[str, str]:
        """Extract instructions with detailed debugging."""
        
        print(f"\n{'='*50}")
        print(f"Debugging {name}")
        print(f"{'='*50}")
        
        try:
            # Create temporary source file
            source_file = os.path.join(self.temp_dir, f"{name}.c")
            binary_file = os.path.join(self.temp_dir, f"{name}.bin")
            
            print(f"1. Writing source to: {source_file}")
            with open(source_file, 'w') as f:
                f.write(source_code)
            
            # Try standard compilation first
            print("2. Attempting standard compilation...")
            compile_cmd = ["gcc", "-O2", "-o", binary_file, source_file]
            print(f"   Command: {' '.join(compile_cmd)}")
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"   Compilation failed!")
                print(f"   stderr: {result.stderr}")
                return None, None
            else:
                print(f"   Compilation successful!")
                
            # Check if binary exists and get size
            if os.path.exists(binary_file):
                size = os.path.getsize(binary_file)
                print(f"   Binary size: {size} bytes")
            else:
                print("   Binary file not found!")
                return None, None
            
            # Try objdump
            print("3. Running objdump...")
            objdump_cmd = ["objdump", "-d", binary_file]
            print(f"   Command: {' '.join(objdump_cmd)}")
            
            objdump_result = subprocess.run(objdump_cmd, capture_output=True, text=True)
            
            if objdump_result.returncode != 0:
                print(f"   objdump failed!")
                print(f"   stderr: {objdump_result.stderr}")
                return None, None
            else:
                print(f"   objdump successful!")
                print(f"   Output length: {len(objdump_result.stdout)} characters")
                
            # Show first few lines of objdump output
            lines = objdump_result.stdout.split('\\n')[:10]
            print("   First 10 lines of objdump output:")
            for i, line in enumerate(lines):
                print(f"     {i}: {line}")
            
            # Try to extract just .text section
            print("4. Trying .text section extraction...")
            text_cmd = ["objdump", "-d", "-j", ".text", binary_file]
            text_result = subprocess.run(text_cmd, capture_output=True, text=True)
            
            if text_result.returncode == 0:
                print(f"   .text extraction successful!")
                print(f"   .text output length: {len(text_result.stdout)} characters")
                
                text_lines = text_result.stdout.split('\\n')[:10]
                print("   First 10 lines of .text output:")
                for i, line in enumerate(text_lines):
                    print(f"     {i}: {line}")
                
                # Use .text output for parsing
                objdump_output = text_result.stdout
            else:
                print(f"   .text extraction failed, using full objdump")
                objdump_output = objdump_result.stdout
            
            # Parse the output
            print("5. Parsing objdump output...")
            opcodes, assembly = self._parse_with_debug(objdump_output)
            
            print(f"6. Results:")
            print(f"   Opcodes: '{opcodes[:100]}...' ({len(opcodes)} chars)")
            print(f"   Assembly: '{assembly[:200]}...' ({len(assembly)} chars)")
            
            return opcodes, assembly
            
        except Exception as e:
            print(f"Error in extraction: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _parse_with_debug(self, objdump_output: str) -> Tuple[str, str]:
        """Parse objdump output with debugging."""
        
        lines = objdump_output.split('\\n')
        print(f"   Total lines to parse: {len(lines)}")
        
        opcodes = []
        assembly_instructions = []
        instruction_count = 0
        
        for i, line in enumerate(lines):
            # Show first few lines being processed
            if i < 5:
                print(f"   Processing line {i}: '{line}'")
            
            # Look for instruction lines
            if ':' in line and '\\t' in line:
                try:
                    # Split by colon
                    parts = line.split(':', 1)
                    if len(parts) < 2:
                        continue
                    
                    instruction_part = parts[1].strip()
                    
                    # Split by tabs
                    tab_parts = instruction_part.split('\\t')
                    if len(tab_parts) < 2:
                        continue
                    
                    # Extract hex bytes
                    hex_part = tab_parts[0].strip()
                    if hex_part:
                        # Remove spaces and keep only hex
                        hex_bytes = re.sub(r'[^0-9a-fA-F]', '', hex_part)
                        if hex_bytes:
                            opcodes.append(hex_bytes)
                            if i < 5:
                                print(f"     Extracted opcode: '{hex_bytes}'")
                    
                    # Extract assembly
                    asm_part = tab_parts[-1].strip()
                    if asm_part:
                        assembly_instructions.append(asm_part)
                        if i < 5:
                            print(f"     Extracted assembly: '{asm_part}'")
                        
                        instruction_count += 1
                        
                except Exception as e:
                    if i < 5:
                        print(f"     Error parsing line: {e}")
                    continue
        
        print(f"   Found {instruction_count} instructions")
        print(f"   Total opcode characters: {sum(len(op) for op in opcodes)}")
        print(f"   Total assembly lines: {len(assembly_instructions)}")
        
        final_opcodes = ''.join(opcodes)
        final_assembly = '\\n'.join(assembly_instructions)
        
        return final_opcodes, final_assembly

def test_simple_program():
    """Test with a very simple program to ensure the process works."""
    
    simple_code = '''
#include <stdio.h>

int main() {
    printf("Hello\\n");
    return 0;
}
'''
    
    extractor = DiagnosticExtractor()
    opcodes, assembly = extractor.extract_and_debug(simple_code, "hello_test")
    
    if opcodes and assembly:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        source_tokens = len(tokenizer.encode(simple_code))
        opcode_tokens = len(tokenizer.encode(opcodes))
        assembly_tokens = len(tokenizer.encode(assembly))
        
        print(f"\\n{'='*50}")
        print(f"TOKEN ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Source code tokens: {source_tokens}")
        print(f"Opcode tokens: {opcode_tokens}")
        print(f"Assembly tokens: {assembly_tokens}")
        
        if opcode_tokens > 0:
            opcode_eff = (source_tokens - opcode_tokens) / source_tokens * 100
            print(f"Opcode efficiency: {opcode_eff:.1f}%")
        
        if assembly_tokens > 0:
            assembly_eff = (source_tokens - assembly_tokens) / source_tokens * 100
            print(f"Assembly efficiency: {assembly_eff:.1f}%")
            
        print(f"\\nRaw opcode string (first 100 chars): '{opcodes[:100]}'")
        print(f"Raw assembly string (first 200 chars): '{assembly[:200]}'")
    
    else:
        print("Failed to extract instructions!")

def test_compilation_environment():
    """Test if the compilation environment is working."""
    
    print("TESTING COMPILATION ENVIRONMENT")
    print("="*40)
    
    # Test GCC
    try:
        result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ GCC is available")
            print(f"  Version: {result.stdout.split()[0]} {result.stdout.split()[3] if len(result.stdout.split()) > 3 else 'unknown'}")
        else:
            print("✗ GCC not working")
    except:
        print("✗ GCC not found")
    
    # Test objdump
    try:
        result = subprocess.run(["objdump", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ objdump is available")
        else:
            print("✗ objdump not working")
    except:
        print("✗ objdump not found")
    
    # Test tiktoken
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        test_tokens = tokenizer.encode("Hello world")
        print(f"✓ tiktoken working (test: {len(test_tokens)} tokens)")
    except:
        print("✗ tiktoken not working")

def main():
    """Run diagnostic analysis."""
    
    print("DIAGNOSTIC TOKEN ANALYSIS")
    print("="*40)
    
    # Test environment first
    test_compilation_environment()
    
    print("\\n")
    
    # Test with simple program
    test_simple_program()

if __name__ == "__main__":
    main()