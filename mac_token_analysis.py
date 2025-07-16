#!/usr/bin/env python3
"""
Working macOS Analysis - Fixed Parser
=====================================

This version properly handles the objdump output format on macOS ARM64.
"""

import os
import subprocess
import tempfile
import tiktoken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import Tuple, Optional, List

class WorkingMacOSExtractor:
    """Working instruction extractor for macOS ARM64."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_instructions(self, source_code: str, name: str) -> Tuple[str, str]:
        """Extract instructions with proper parsing."""
        
        try:
            # Create files
            source_file = os.path.join(self.temp_dir, f"{name}.c")
            binary_file = os.path.join(self.temp_dir, f"{name}.bin")
            
            with open(source_file, 'w') as f:
                f.write(source_code)
            
            # Compile
            compile_cmd = ["gcc", "-O2", "-o", binary_file, source_file]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Compilation failed for {name}")
                return None, None
            
            # Get objdump output
            objdump_cmd = ["objdump", "-d", binary_file]
            objdump_result = subprocess.run(objdump_cmd, capture_output=True, text=True)
            
            if objdump_result.returncode != 0:
                print(f"objdump failed for {name}")
                return None, None
            
            # Parse the output properly
            opcodes, assembly = self._parse_objdump_output(objdump_result.stdout)
            
            print(f"  Extracted {len(opcodes)} opcode chars, {len(assembly)} assembly chars")
            if opcodes:
                print(f"  Opcodes sample: {opcodes[:50]}...")
            if assembly:
                print(f"  Assembly sample: {assembly[:100]}...")
            
            return opcodes, assembly
            
        except Exception as e:
            print(f"Error extracting instructions for {name}: {e}")
            return None, None
    
    def _parse_objdump_output(self, output: str) -> Tuple[str, str]:
        """Parse objdump output correctly."""
        
        opcodes = []
        assembly_instructions = []
        
        # The key insight: we need to split by actual newlines in the content
        # First, let's look for instruction patterns in the entire output
        
        # Pattern for ARM64 instructions: address: hex_bytes    mnemonic operands
        instruction_pattern = re.compile(r'([0-9a-f]+):\s+([0-9a-f]+)\s+([^\\n]+)', re.IGNORECASE)
        
        matches = instruction_pattern.findall(output)
        
        for address, hex_bytes, asm_instruction in matches:
            # Only include if it looks like a real instruction (not data)
            if len(hex_bytes) == 8:  # ARM64 instructions are 4 bytes = 8 hex chars
                opcodes.append(hex_bytes)
                assembly_instructions.append(asm_instruction.strip())
        
        # Alternative approach: split by actual newlines and parse line by line
        if not opcodes:  # If regex didn't work, try line-by-line
            lines = output.split('\\n')
            for line in lines:
                line = line.strip()
                if ':' in line and len(line) > 10:
                    # Look for pattern: hex_address: hex_bytes instruction
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        addr_part = parts[0].strip()
                        rest = parts[1].strip()
                        
                        # Check if address is hex
                        if re.match(r'^[0-9a-fA-F]+$', addr_part):
                            # Split the rest into hex bytes and assembly
                            match = re.match(r'^([0-9a-fA-F]+)\\s+(.+)$', rest)
                            if match:
                                hex_bytes = match.group(1)
                                asm_instruction = match.group(2).strip()
                                
                                # Only 4-byte instructions
                                if len(hex_bytes) == 8:
                                    opcodes.append(hex_bytes)
                                    assembly_instructions.append(asm_instruction)
        
        return ''.join(opcodes), '\\n'.join(assembly_instructions)
    
    def __del__(self):
        """Cleanup."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class WorkingTokenAnalyzer:
    """Working token analyzer."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
        self.extractor = WorkingMacOSExtractor()
    
    def analyze_samples(self, samples: List[Tuple[str, str, int]]) -> pd.DataFrame:
        """Analyze samples with working parser."""
        
        results = []
        
        for name, source_code, estimated_lines in samples:
            print(f"\\nAnalyzing {name} (~{estimated_lines} lines)...")
            
            # Source tokens
            source_tokens = len(self.tokenizer.encode(source_code))
            print(f"  Source tokens: {source_tokens}")
            
            # Extract instructions
            opcodes, assembly = self.extractor.extract_instructions(source_code, name)
            
            if opcodes is None or assembly is None:
                print(f"  ❌ Failed to extract instructions")
                continue
            
            if not opcodes and not assembly:
                print(f"  ❌ No instructions extracted")
                continue
            
            # Count tokens
            opcode_tokens = len(self.tokenizer.encode(opcodes)) if opcodes else 0
            assembly_tokens = len(self.tokenizer.encode(assembly)) if assembly else 0
            
            print(f"  Opcode tokens: {opcode_tokens}")
            print(f"  Assembly tokens: {assembly_tokens}")
            
            # Calculate efficiency
            opcode_efficiency = (source_tokens - opcode_tokens) / source_tokens * 100 if source_tokens > 0 else 0
            assembly_efficiency = (source_tokens - assembly_tokens) / source_tokens * 100 if source_tokens > 0 else 0
            
            print(f"  Opcode efficiency: {opcode_efficiency:.1f}%")
            print(f"  Assembly efficiency: {assembly_efficiency:.1f}%")
            
            results.append({
                'name': name,
                'estimated_lines': estimated_lines,
                'source_tokens': source_tokens,
                'opcode_tokens': opcode_tokens,
                'assembly_tokens': assembly_tokens,
                'opcode_efficiency_pct': opcode_efficiency,
                'assembly_efficiency_pct': assembly_efficiency,
                'opcode_chars': len(opcodes),
                'assembly_chars': len(assembly),
                'scale_category': self._categorize_scale(estimated_lines)
            })
        
        return pd.DataFrame(results)
    
    def _categorize_scale(self, lines: int) -> str:
        if lines < 50:
            return 'Small'
        elif lines < 150:
            return 'Medium'
        elif lines < 400:
            return 'Large'
        else:
            return 'Very Large'
    
    def create_analysis_report(self, df: pd.DataFrame):
        """Create comprehensive analysis report."""
        
        if df.empty:
            print("No data to analyze.")
            return
        
        print("\\n" + "="*60)
        print("WORKING ANALYSIS RESULTS")
        print("="*60)
        
        # Overall stats
        print(f"Successfully analyzed: {len(df)} samples")
        print(f"Source tokens range: {df['source_tokens'].min()} - {df['source_tokens'].max()}")
        print(f"Average source tokens: {df['source_tokens'].mean():.1f}")
        
        # Efficiency stats
        print(f"\\nOPCODE EFFICIENCY:")
        print(f"  Range: {df['opcode_efficiency_pct'].min():.1f}% to {df['opcode_efficiency_pct'].max():.1f}%")
        print(f"  Average: {df['opcode_efficiency_pct'].mean():.1f}%")
        
        print(f"\\nASSEMBLY EFFICIENCY:")
        print(f"  Range: {df['assembly_efficiency_pct'].min():.1f}% to {df['assembly_efficiency_pct'].max():.1f}%")
        print(f"  Average: {df['assembly_efficiency_pct'].mean():.1f}%")
        
        # Best performers
        if df['opcode_efficiency_pct'].max() > 0:
            best_opcode = df.loc[df['opcode_efficiency_pct'].idxmax()]
            print(f"\\nBest opcode efficiency: {best_opcode['opcode_efficiency_pct']:.1f}% ({best_opcode['name']})")
        
        if df['assembly_efficiency_pct'].max() > 0:
            best_assembly = df.loc[df['assembly_efficiency_pct'].idxmax()]
            print(f"Best assembly efficiency: {best_assembly['assembly_efficiency_pct']:.1f}% ({best_assembly['name']})")
        
        # Scale analysis
        print(f"\\nBY SCALE CATEGORY:")
        scale_stats = df.groupby('scale_category').agg({
            'source_tokens': 'mean',
            'opcode_efficiency_pct': 'mean',
            'assembly_efficiency_pct': 'mean'
        }).round(1)
        print(scale_stats)
        
        # Detailed results
        print(f"\\nDETAILED RESULTS:")
        print(df[['name', 'estimated_lines', 'source_tokens', 'opcode_tokens', 'assembly_tokens', 
                 'opcode_efficiency_pct', 'assembly_efficiency_pct']].to_string(index=False))
        
        # Create visualization
        self._create_visualization(df)
    
    def _create_visualization(self, df: pd.DataFrame):
        """Create visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Efficiency vs size
        ax1 = axes[0, 0]
        ax1.scatter(df['estimated_lines'], df['opcode_efficiency_pct'], 
                   label='Opcodes', s=100, alpha=0.7)
        ax1.scatter(df['estimated_lines'], df['assembly_efficiency_pct'], 
                   label='Assembly', s=100, alpha=0.7)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Program Size (Lines)')
        ax1.set_ylabel('Efficiency (%)')
        ax1.set_title('Token Efficiency vs Program Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Token counts
        ax2 = axes[0, 1]
        x = range(len(df))
        width = 0.25
        
        ax2.bar([i - width for i in x], df['source_tokens'], width, 
                label='Source', alpha=0.8)
        ax2.bar([i for i in x], df['opcode_tokens'], width, 
                label='Opcodes', alpha=0.8)
        ax2.bar([i + width for i in x], df['assembly_tokens'], width, 
                label='Assembly', alpha=0.8)
        
        ax2.set_xlabel('Programs')
        ax2.set_ylabel('Token Count')
        ax2.set_title('Token Counts Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['name'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency by scale
        ax3 = axes[1, 0]
        scale_data = df.groupby('scale_category')[['opcode_efficiency_pct', 'assembly_efficiency_pct']].mean()
        scale_data.plot(kind='bar', ax=ax3)
        ax3.set_xlabel('Scale Category')
        ax3.set_ylabel('Average Efficiency (%)')
        ax3.set_title('Efficiency by Scale Category')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Character efficiency
        ax4 = axes[1, 1]
        ax4.scatter(df['opcode_chars'], df['opcode_tokens'], 
                   label='Opcodes', s=100, alpha=0.7)
        ax4.scatter(df['assembly_chars'], df['assembly_tokens'], 
                   label='Assembly', s=100, alpha=0.7)
        ax4.set_xlabel('Character Count')
        ax4.set_ylabel('Token Count')
        ax4.set_title('Characters vs Tokens')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('working_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def get_test_samples() -> List[Tuple[str, str, int]]:
    """Get test samples."""
    
    return [
        ("tiny_program", '''
#include <stdio.h>
int main() {
    return 42;
}
''', 5),
        
        ("hello_world", '''
#include <stdio.h>
int main() {
    printf("Hello, World!\\n");
    return 0;
}
''', 10),
        
        ("simple_math", '''
#include <stdio.h>
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
int main() {
    int x = 5, y = 3;
    printf("Add: %d\\n", add(x, y));
    printf("Multiply: %d\\n", multiply(x, y));
    return 0;
}
''', 20),
        
        ("loop_example", '''
#include <stdio.h>
int main() {
    int sum = 0;
    for (int i = 1; i <= 10; i++) {
        sum += i;
    }
    printf("Sum: %d\\n", sum);
    
    int factorial = 1;
    for (int i = 1; i <= 5; i++) {
        factorial *= i;
    }
    printf("Factorial: %d\\n", factorial);
    return 0;
}
''', 40),
        
        ("array_sorting", '''
#include <stdio.h>
#define SIZE 10

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\\n");
}

int main() {
    int arr[SIZE] = {64, 34, 25, 12, 22, 11, 90, 5, 77, 30};
    
    printf("Original: ");
    printArray(arr, SIZE);
    
    bubbleSort(arr, SIZE);
    
    printf("Sorted: ");
    printArray(arr, SIZE);
    
    return 0;
}
''', 80),
        
        ("string_processing", '''
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int stringLength(char* str) {
    int len = 0;
    while (str[len]) len++;
    return len;
}

void reverseString(char* str) {
    int len = stringLength(str);
    for (int i = 0; i < len/2; i++) {
        char temp = str[i];
        str[i] = str[len-1-i];
        str[len-1-i] = temp;
    }
}

int isPalindrome(char* str) {
    int len = stringLength(str);
    for (int i = 0; i < len/2; i++) {
        if (str[i] != str[len-1-i]) return 0;
    }
    return 1;
}

char* concatenate(char* s1, char* s2) {
    int len1 = stringLength(s1);
    int len2 = stringLength(s2);
    char* result = malloc(len1 + len2 + 1);
    
    for (int i = 0; i < len1; i++) {
        result[i] = s1[i];
    }
    for (int i = 0; i < len2; i++) {
        result[len1 + i] = s2[i];
    }
    result[len1 + len2] = '\\0';
    return result;
}

int main() {
    char str1[] = "hello";
    char str2[] = "world";
    
    printf("Length of '%s': %d\\n", str1, stringLength(str1));
    
    char test[] = "racecar";
    printf("'%s' palindrome: %s\\n", test, isPalindrome(test) ? "yes" : "no");
    
    char* combined = concatenate(str1, str2);
    printf("Combined: %s\\n", combined);
    
    reverseString(combined);
    printf("Reversed: %s\\n", combined);
    
    free(combined);
    return 0;
}
''', 150),
    ]

def main():
    """Run the working analysis."""
    
    print("WORKING MACOS ARM64 TOKEN ANALYSIS")
    print("="*50)
    
    # Get samples
    samples = get_test_samples()
    
    # Run analysis
    analyzer = WorkingTokenAnalyzer("gpt-4")
    results_df = analyzer.analyze_samples(samples)
    
    # Generate report
    analyzer.create_analysis_report(results_df)
    
    # Save results
    if not results_df.empty:
        results_df.to_csv('working_analysis_results.csv', index=False)
        print(f"\\nResults saved to 'working_analysis_results.csv'")
    
    print("\\nAnalysis complete!")

if __name__ == "__main__":
    main()