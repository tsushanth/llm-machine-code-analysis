#!/usr/bin/env python3
"""
macOS ARM64 Fixed Analysis
=========================

This version is specifically designed for macOS ARM64 systems and properly
handles the mach-o format and objdump output.
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

class MacOSInstructionExtractor:
    """Instruction extractor optimized for macOS ARM64."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_instructions(self, source_code: str, name: str) -> Tuple[str, str]:
        """Extract instructions for macOS ARM64."""
        
        try:
            # Create temporary files
            source_file = os.path.join(self.temp_dir, f"{name}.c")
            binary_file = os.path.join(self.temp_dir, f"{name}.bin")
            
            with open(source_file, 'w') as f:
                f.write(source_code)
            
            # Compile with standard flags
            compile_cmd = ["gcc", "-O2", "-o", binary_file, source_file]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Compilation failed for {name}: {result.stderr}")
                return None, None
            
            # Use objdump to get disassembly
            objdump_cmd = ["objdump", "-d", binary_file]
            objdump_result = subprocess.run(objdump_cmd, capture_output=True, text=True)
            
            if objdump_result.returncode != 0:
                print(f"objdump failed for {name}: {objdump_result.stderr}")
                return None, None
            
            # Parse the ARM64 mach-o format
            opcodes, assembly = self._parse_macos_objdump(objdump_result.stdout)
            
            return opcodes, assembly
            
        except Exception as e:
            print(f"Error extracting instructions for {name}: {e}")
            return None, None
    
    def _parse_macos_objdump(self, objdump_output: str) -> Tuple[str, str]:
        """Parse macOS objdump output for ARM64."""
        
        lines = objdump_output.split('\\n')
        opcodes = []
        assembly_instructions = []
        
        # Look for instruction lines in ARM64 format
        # Format: "100003f74: a9bf7bfd    stp x29, x30, [sp, #-0x10]!"
        
        for line in lines:
            # ARM64 instruction lines have: address: hex_bytes    mnemonic
            if ':' in line and len(line.strip()) > 0:
                # Remove leading/trailing whitespace
                line = line.strip()
                
                # Split by colon
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                
                address_part = parts[0].strip()
                instruction_part = parts[1].strip()
                
                # Check if address looks like a hex address
                if not re.match(r'^[0-9a-fA-F]+$', address_part):
                    continue
                
                # Parse instruction part
                # Format: "a9bf7bfd    stp x29, x30, [sp, #-0x10]!"
                instruction_match = re.match(r'^([0-9a-fA-F]+)\\s+(.+)$', instruction_part)
                
                if instruction_match:
                    hex_bytes = instruction_match.group(1)
                    asm_instruction = instruction_match.group(2).strip()
                    
                    # Add to our collections
                    opcodes.append(hex_bytes)
                    assembly_instructions.append(asm_instruction)
        
        return ''.join(opcodes), '\\n'.join(assembly_instructions)
    
    def __del__(self):
        """Cleanup temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class MacOSTokenAnalyzer:
    """Token analyzer for macOS ARM64."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
        self.extractor = MacOSInstructionExtractor()
    
    def analyze_samples(self, samples: List[Tuple[str, str, int]]) -> pd.DataFrame:
        """Analyze token efficiency for macOS ARM64."""
        
        results = []
        
        for name, source_code, estimated_lines in samples:
            print(f"Analyzing {name} (~{estimated_lines} lines)...")
            
            # Count source tokens
            source_tokens = len(self.tokenizer.encode(source_code))
            
            # Extract instructions
            opcodes, assembly = self.extractor.extract_instructions(source_code, name)
            
            if opcodes is None or assembly is None:
                print(f"  Failed to extract instructions for {name}")
                continue
            
            # Debug: show what we extracted
            print(f"  Extracted {len(opcodes)} opcode chars, {len(assembly)} assembly chars")
            print(f"  Opcodes sample: {opcodes[:50]}...")
            print(f"  Assembly sample: {assembly[:100]}...")
            
            # Count tokens for extracted content
            opcode_tokens = len(self.tokenizer.encode(opcodes)) if opcodes else 0
            assembly_tokens = len(self.tokenizer.encode(assembly)) if assembly else 0
            
            # Calculate efficiency
            opcode_efficiency = (source_tokens - opcode_tokens) / source_tokens * 100 if source_tokens > 0 else 0
            assembly_efficiency = (source_tokens - assembly_tokens) / source_tokens * 100 if source_tokens > 0 else 0
            
            results.append({
                'name': name,
                'estimated_lines': estimated_lines,
                'source_tokens': source_tokens,
                'source_chars': len(source_code),
                'opcode_tokens': opcode_tokens,
                'assembly_tokens': assembly_tokens,
                'opcode_chars': len(opcodes) if opcodes else 0,
                'assembly_chars': len(assembly) if assembly else 0,
                'opcode_efficiency_pct': opcode_efficiency,
                'assembly_efficiency_pct': assembly_efficiency,
                'scale_category': self._categorize_scale(estimated_lines)
            })
        
        return pd.DataFrame(results)
    
    def _categorize_scale(self, lines: int) -> str:
        """Categorize code samples by scale."""
        if lines < 100:
            return 'Small'
        elif lines < 300:
            return 'Medium'
        elif lines < 800:
            return 'Large'
        else:
            return 'Very Large'
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualizations for the analysis."""
        
        if df.empty:
            print("No data to visualize.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Efficiency vs program size
        ax1 = axes[0, 0]
        ax1.scatter(df['estimated_lines'], df['opcode_efficiency_pct'], 
                   label='Opcodes', alpha=0.7, s=100)
        ax1.scatter(df['estimated_lines'], df['assembly_efficiency_pct'], 
                   label='Assembly', alpha=0.7, s=100)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Program Size (Lines)')
        ax1.set_ylabel('Token Efficiency (%)')
        ax1.set_title('Token Efficiency vs Program Size (macOS ARM64)')
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
        ax2.set_title('Token Counts by Program')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['name'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Character vs token ratios
        ax3 = axes[1, 0]
        ax3.scatter(df['source_chars'], df['source_tokens'], 
                   label='Source', alpha=0.7, s=100)
        ax3.scatter(df['opcode_chars'], df['opcode_tokens'], 
                   label='Opcodes', alpha=0.7, s=100)
        ax3.scatter(df['assembly_chars'], df['assembly_tokens'], 
                   label='Assembly', alpha=0.7, s=100)
        
        ax3.set_xlabel('Character Count')
        ax3.set_ylabel('Token Count')
        ax3.set_title('Characters vs Tokens')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency by scale category
        ax4 = axes[1, 1]
        if len(df['scale_category'].unique()) > 1:
            scale_stats = df.groupby('scale_category')[['opcode_efficiency_pct', 'assembly_efficiency_pct']].mean()
            scale_stats.plot(kind='bar', ax=ax4)
            ax4.set_xlabel('Scale Category')
            ax4.set_ylabel('Average Efficiency (%)')
            ax4.set_title('Efficiency by Scale Category')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('macos_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        self._print_summary(df)
    
    def _print_summary(self, df: pd.DataFrame):
        """Print analysis summary."""
        
        print("\\n" + "="*60)
        print("MACOS ARM64 ANALYSIS SUMMARY")
        print("="*60)
        
        if df.empty:
            print("No successful analyses.")
            return
        
        print(f"Successfully analyzed: {len(df)} samples")
        print(f"Average source tokens: {df['source_tokens'].mean():.1f}")
        print(f"Average opcode tokens: {df['opcode_tokens'].mean():.1f}")
        print(f"Average assembly tokens: {df['assembly_tokens'].mean():.1f}")
        
        print(f"\\nEfficiency Results:")
        print(f"Best opcode efficiency: {df['opcode_efficiency_pct'].max():.1f}%")
        print(f"Best assembly efficiency: {df['assembly_efficiency_pct'].max():.1f}%")
        print(f"Average opcode efficiency: {df['opcode_efficiency_pct'].mean():.1f}%")
        print(f"Average assembly efficiency: {df['assembly_efficiency_pct'].mean():.1f}%")
        
        # Check for positive efficiency
        positive_opcode = df[df['opcode_efficiency_pct'] > 0]
        positive_assembly = df[df['assembly_efficiency_pct'] > 0]
        
        if not positive_opcode.empty:
            print(f"\\nOpcode efficiency > 0: {len(positive_opcode)} samples")
            best_opcode = positive_opcode.loc[positive_opcode['opcode_efficiency_pct'].idxmax()]
            print(f"Best: {best_opcode['opcode_efficiency_pct']:.1f}% ({best_opcode['name']})")
        
        if not positive_assembly.empty:
            print(f"Assembly efficiency > 0: {len(positive_assembly)} samples")
            best_assembly = positive_assembly.loc[positive_assembly['assembly_efficiency_pct'].idxmax()]
            print(f"Best: {best_assembly['assembly_efficiency_pct']:.1f}% ({best_assembly['name']})")

def get_test_samples() -> List[Tuple[str, str, int]]:
    """Get test samples for analysis."""
    
    return [
        ("hello_world", '''
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
''', 10),
        
        ("simple_math", '''
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

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
    int n = 5;
    for (int i = 1; i <= n; i++) {
        factorial *= i;
    }
    printf("Factorial: %d\\n", factorial);
    
    return 0;
}
''', 30),
        
        ("array_operations", '''
#include <stdio.h>

#define SIZE 10

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\\n");
}

int findMax(int arr[], int size) {
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

void bubbleSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int numbers[SIZE] = {64, 34, 25, 12, 22, 11, 90, 5, 77, 30};
    
    printf("Original array: ");
    printArray(numbers, SIZE);
    
    printf("Maximum: %d\\n", findMax(numbers, SIZE));
    
    bubbleSort(numbers, SIZE);
    printf("Sorted array: ");
    printArray(numbers, SIZE);
    
    return 0;
}
''', 60),
        
        ("string_functions", '''
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int stringLength(char* str) {
    int length = 0;
    while (str[length] != '\\0') {
        length++;
    }
    return length;
}

void reverseString(char* str) {
    int length = stringLength(str);
    for (int i = 0; i < length / 2; i++) {
        char temp = str[i];
        str[i] = str[length - 1 - i];
        str[length - 1 - i] = temp;
    }
}

int isPalindrome(char* str) {
    int length = stringLength(str);
    for (int i = 0; i < length / 2; i++) {
        if (str[i] != str[length - 1 - i]) {
            return 0;
        }
    }
    return 1;
}

char* concatenate(char* str1, char* str2) {
    int len1 = stringLength(str1);
    int len2 = stringLength(str2);
    char* result = (char*)malloc(len1 + len2 + 1);
    
    int i, j;
    for (i = 0; i < len1; i++) {
        result[i] = str1[i];
    }
    for (j = 0; j < len2; j++) {
        result[i + j] = str2[j];
    }
    result[i + j] = '\\0';
    
    return result;
}

int main() {
    char str1[] = "hello";
    char str2[] = "world";
    
    printf("Length of '%s': %d\\n", str1, stringLength(str1));
    
    char testStr[] = "racecar";
    printf("'%s' is palindrome: %s\\n", testStr, isPalindrome(testStr) ? "Yes" : "No");
    
    char* combined = concatenate(str1, str2);
    printf("Concatenated: %s\\n", combined);
    
    reverseString(combined);
    printf("Reversed: %s\\n", combined);
    
    free(combined);
    return 0;
}
''', 100),
    ]

def main():
    """Run the macOS ARM64 analysis."""
    
    print("MACOS ARM64 TOKEN EFFICIENCY ANALYSIS")
    print("="*45)
    
    # Get test samples
    samples = get_test_samples()
    
    # Initialize analyzer
    analyzer = MacOSTokenAnalyzer("gpt-4")
    
    # Run analysis
    results_df = analyzer.analyze_samples(samples)
    
    if not results_df.empty:
        # Display results
        print("\\nResults:")
        print(results_df[['name', 'estimated_lines', 'source_tokens', 
                         'opcode_tokens', 'assembly_tokens', 'opcode_efficiency_pct', 
                         'assembly_efficiency_pct']].to_string(index=False))
        
        # Create visualizations
        analyzer.create_visualizations(results_df)
        
        # Save results
        results_df.to_csv('macos_analysis_results.csv', index=False)
        print(f"\\nResults saved to 'macos_analysis_results.csv'")
        
    else:
        print("No successful analyses to display.")

if __name__ == "__main__":
    main()