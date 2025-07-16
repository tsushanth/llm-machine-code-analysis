#!/usr/bin/env python3
"""
LLM Token Cost Analysis: Source Code vs Machine Code
===================================================

This project analyzes potential token cost savings when feeding machine code
instead of high-level programming language code to Large Language Models.

Requirements:
- pip install tiktoken transformers datasets numpy matplotlib seaborn pandas
- C compiler (gcc) for generating machine code
- Optional: capstone for disassembly analysis

Author: Research Project
Date: July 2025
"""

import os
import subprocess
import tempfile
import tiktoken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import hashlib

class TokenAnalyzer:
    """Analyzes token counts for different representations of code."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize with specified tokenizer model."""
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in given text."""
        return len(self.tokenizer.encode(text))
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze token statistics for text."""
        tokens = self.tokenizer.encode(text)
        return {
            'token_count': len(tokens),
            'char_count': len(text),
            'chars_per_token': len(text) / len(tokens) if tokens else 0,
            'text_length': len(text)
        }

class CodeCompiler:
    """Handles compilation of source code to machine code."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def compile_c_code(self, source_code: str) -> Optional[Tuple[str, str]]:
        """
        Compile C code and return both binary and assembly representations.
        Returns: (binary_hex, assembly_text) or None if compilation fails
        """
        try:
            # Create temporary source file
            source_file = os.path.join(self.temp_dir, "temp.c")
            binary_file = os.path.join(self.temp_dir, "temp.bin")
            
            with open(source_file, 'w') as f:
                f.write(source_code)
            
            # Compile to binary
            compile_cmd = ["gcc", "-O2", "-o", binary_file, source_file]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                return None
            
            # Get binary as hex string
            with open(binary_file, 'rb') as f:
                binary_data = f.read()
                binary_hex = binary_data.hex()
            
            # Get assembly disassembly
            disasm_cmd = ["objdump", "-d", binary_file]
            disasm_result = subprocess.run(disasm_cmd, capture_output=True, text=True)
            
            if disasm_result.returncode != 0:
                assembly_text = "Disassembly failed"
            else:
                assembly_text = disasm_result.stdout
            
            return binary_hex, assembly_text
            
        except Exception as e:
            print(f"Error compiling code: {e}")
            return None
    
    def __del__(self):
        """Cleanup temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class CodeSampleGenerator:
    """Generates various code samples for analysis."""
    
    @staticmethod
    def get_sample_programs() -> List[Tuple[str, str]]:
        """Return list of (name, source_code) tuples."""
        return [
            ("hello_world", """
#include <stdio.h>
int main() {
    printf("Hello, World!\\n");
    return 0;
}
"""),
            ("fibonacci", """
#include <stdio.h>
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}
int main() {
    int n = 10;
    printf("Fibonacci(%d) = %d\\n", n, fibonacci(n));
    return 0;
}
"""),
            ("bubble_sort", """
#include <stdio.h>
void bubble_sort(int arr[], int n) {
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
int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr)/sizeof(arr[0]);
    bubble_sort(arr, n);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    return 0;
}
"""),
            ("linked_list", """
#include <stdio.h>
#include <stdlib.h>
struct Node {
    int data;
    struct Node* next;
};
void printList(struct Node* node) {
    while (node != NULL) {
        printf("%d ", node->data);
        node = node->next;
    }
}
int main() {
    struct Node* head = NULL;
    struct Node* second = NULL;
    struct Node* third = NULL;
    head = (struct Node*)malloc(sizeof(struct Node));
    second = (struct Node*)malloc(sizeof(struct Node));
    third = (struct Node*)malloc(sizeof(struct Node));
    head->data = 1;
    head->next = second;
    second->data = 2;
    second->next = third;
    third->data = 3;
    third->next = NULL;
    printList(head);
    return 0;
}
"""),
            ("simple_calculator", """
#include <stdio.h>
float calculate(float a, float b, char op) {
    switch(op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return b != 0 ? a / b : 0;
        default: return 0;
    }
}
int main() {
    float a = 10.5, b = 3.2;
    char ops[] = {'+', '-', '*', '/'};
    for (int i = 0; i < 4; i++) {
        printf("%.2f %c %.2f = %.2f\\n", a, ops[i], b, calculate(a, b, ops[i]));
    }
    return 0;
}
""")
        ]

class CostAnalysisRunner:
    """Main class to run the complete analysis."""
    
    def __init__(self):
        self.analyzer = TokenAnalyzer()
        self.compiler = CodeCompiler()
        self.sample_generator = CodeSampleGenerator()
        self.results = []
    
    def analyze_single_program(self, name: str, source_code: str) -> Dict:
        """Analyze a single program and return results."""
        print(f"Analyzing: {name}")
        
        # Analyze source code
        source_analysis = self.analyzer.analyze_text(source_code)
        
        # Compile and analyze machine code
        compile_result = self.compiler.compile_c_code(source_code)
        
        if compile_result is None:
            return {
                'name': name,
                'source_tokens': source_analysis['token_count'],
                'binary_tokens': 0,
                'assembly_tokens': 0,
                'compilation_success': False,
                'token_savings_binary': 0,
                'token_savings_assembly': 0,
                'savings_percentage_binary': 0,
                'savings_percentage_assembly': 0
            }
        
        binary_hex, assembly_text = compile_result
        
        # Analyze binary representation
        binary_analysis = self.analyzer.analyze_text(binary_hex)
        
        # Analyze assembly representation
        assembly_analysis = self.analyzer.analyze_text(assembly_text)
        
        # Calculate savings
        source_tokens = source_analysis['token_count']
        binary_tokens = binary_analysis['token_count']
        assembly_tokens = assembly_analysis['token_count']
        
        binary_savings = source_tokens - binary_tokens
        assembly_savings = source_tokens - assembly_tokens
        
        binary_savings_pct = (binary_savings / source_tokens * 100) if source_tokens > 0 else 0
        assembly_savings_pct = (assembly_savings / source_tokens * 100) if source_tokens > 0 else 0
        
        return {
            'name': name,
            'source_tokens': source_tokens,
            'binary_tokens': binary_tokens,
            'assembly_tokens': assembly_tokens,
            'compilation_success': True,
            'token_savings_binary': binary_savings,
            'token_savings_assembly': assembly_savings,
            'savings_percentage_binary': binary_savings_pct,
            'savings_percentage_assembly': assembly_savings_pct,
            'source_chars': source_analysis['char_count'],
            'binary_chars': binary_analysis['char_count'],
            'assembly_chars': assembly_analysis['char_count']
        }
    
    def run_full_analysis(self) -> pd.DataFrame:
        """Run analysis on all sample programs."""
        print("Starting comprehensive token cost analysis...")
        
        samples = self.sample_generator.get_sample_programs()
        
        for name, source_code in samples:
            result = self.analyze_single_program(name, source_code)
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive analysis report."""
        successful_analyses = df[df['compilation_success'] == True]
        
        if len(successful_analyses) == 0:
            return "No successful compilations to analyze."
        
        report = f"""
TOKEN COST ANALYSIS REPORT
=========================

Model: {self.analyzer.model_name}
Samples Analyzed: {len(df)}
Successful Compilations: {len(successful_analyses)}

SUMMARY STATISTICS:
------------------
Average Token Counts:
- Source Code: {successful_analyses['source_tokens'].mean():.1f}
- Binary (Hex): {successful_analyses['binary_tokens'].mean():.1f}
- Assembly: {successful_analyses['assembly_tokens'].mean():.1f}

Token Savings (Binary vs Source):
- Average Savings: {successful_analyses['token_savings_binary'].mean():.1f} tokens
- Average Savings Percentage: {successful_analyses['savings_percentage_binary'].mean():.1f}%
- Best Case: {successful_analyses['savings_percentage_binary'].max():.1f}%
- Worst Case: {successful_analyses['savings_percentage_binary'].min():.1f}%

Token Savings (Assembly vs Source):
- Average Savings: {successful_analyses['token_savings_assembly'].mean():.1f} tokens
- Average Savings Percentage: {successful_analyses['savings_percentage_assembly'].mean():.1f}%
- Best Case: {successful_analyses['savings_percentage_assembly'].max():.1f}%
- Worst Case: {successful_analyses['savings_percentage_assembly'].min():.1f}%

DETAILED RESULTS:
----------------
"""
        
        for _, row in successful_analyses.iterrows():
            report += f"""
{row['name'].upper()}:
- Source: {row['source_tokens']} tokens
- Binary: {row['binary_tokens']} tokens ({row['savings_percentage_binary']:+.1f}%)
- Assembly: {row['assembly_tokens']} tokens ({row['savings_percentage_assembly']:+.1f}%)
"""
        
        return report
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualizations of the analysis results."""
        successful_df = df[df['compilation_success'] == True]
        
        if len(successful_df) == 0:
            print("No successful compilations to visualize.")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Token counts comparison
        ax1 = axes[0, 0]
        x = range(len(successful_df))
        width = 0.25
        
        ax1.bar([i - width for i in x], successful_df['source_tokens'], width, 
                label='Source Code', alpha=0.8)
        ax1.bar([i for i in x], successful_df['binary_tokens'], width, 
                label='Binary (Hex)', alpha=0.8)
        ax1.bar([i + width for i in x], successful_df['assembly_tokens'], width, 
                label='Assembly', alpha=0.8)
        
        ax1.set_xlabel('Programs')
        ax1.set_ylabel('Token Count')
        ax1.set_title('Token Counts by Representation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(successful_df['name'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Savings percentages
        ax2 = axes[0, 1]
        ax2.bar(successful_df['name'], successful_df['savings_percentage_binary'], 
                alpha=0.8, label='Binary vs Source')
        ax2.bar(successful_df['name'], successful_df['savings_percentage_assembly'], 
                alpha=0.8, label='Assembly vs Source')
        ax2.set_xlabel('Programs')
        ax2.set_ylabel('Savings Percentage (%)')
        ax2.set_title('Token Savings Percentages')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot: Source tokens vs Binary savings
        ax3 = axes[1, 0]
        ax3.scatter(successful_df['source_tokens'], successful_df['savings_percentage_binary'], 
                   alpha=0.7, s=100)
        ax3.set_xlabel('Source Code Tokens')
        ax3.set_ylabel('Binary Savings (%)')
        ax3.set_title('Source Size vs Binary Savings')
        ax3.grid(True, alpha=0.3)
        
        # Add labels for each point
        for i, row in successful_df.iterrows():
            ax3.annotate(row['name'], 
                        (row['source_tokens'], row['savings_percentage_binary']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Distribution of savings
        ax4 = axes[1, 1]
        ax4.hist(successful_df['savings_percentage_binary'], bins=10, alpha=0.7, 
                label='Binary Savings', edgecolor='black')
        ax4.hist(successful_df['savings_percentage_assembly'], bins=10, alpha=0.7, 
                label='Assembly Savings', edgecolor='black')
        ax4.set_xlabel('Savings Percentage (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Token Savings')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('token_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, df: pd.DataFrame, filename: str = "token_analysis_results.csv"):
        """Save results to CSV file."""
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    """Main function to run the analysis."""
    print("LLM Token Cost Analysis: Source Code vs Machine Code")
    print("=" * 55)
    
    # Initialize analyzer
    analyzer = CostAnalysisRunner()
    
    # Run analysis
    results_df = analyzer.run_full_analysis()
    
    # Generate and display report
    report = analyzer.generate_report(results_df)
    print(report)
    
    # Create visualizations
    analyzer.create_visualizations(results_df)
    
    # Save results
    analyzer.save_results(results_df)
    
    print("\nAnalysis complete! Check 'token_analysis_results.png' for visualizations.")
    print("Raw data saved to 'token_analysis_results.csv'")

if __name__ == "__main__":
    main()