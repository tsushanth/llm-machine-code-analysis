#!/usr/bin/env python3
"""
Comprehensive Token Cost Analysis Script
========================================

This script performs a complete analysis of token costs when using machine code
vs source code for LLM prompts, including realistic prompt scenarios.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime
import argparse

# Import our custom modules (assuming they're in the same directory)
from llm_token_analysis import CostAnalysisRunner, TokenAnalyzer, CodeCompiler
from config_and_samples import (
    ExtendedSampleGenerator, PromptTemplateGenerator, 
    CostCalculator, AnalysisConfig, load_config
)

class ComprehensiveAnalyzer:
    """Performs comprehensive analysis across multiple scenarios."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.token_analyzer = TokenAnalyzer(config.model_name)
        self.compiler = CodeCompiler()
        self.prompt_generator = PromptTemplateGenerator()
        self.cost_calculator = CostCalculator()
        self.results = []
        
    def analyze_prompt_scenario(self, code_name: str, source_code: str, 
                              prompt_type: str) -> Dict:
        """Analyze a specific prompt scenario."""
        
        # Generate the appropriate prompt
        if prompt_type == "explanation":
            prompt = self.prompt_generator.create_code_explanation_prompt(source_code)
        elif prompt_type == "optimization":
            prompt = self.prompt_generator.create_code_optimization_prompt(source_code)
        elif prompt_type == "debugging":
            prompt = self.prompt_generator.create_code_debugging_prompt(source_code)
        elif prompt_type == "conversion":
            prompt = self.prompt_generator.create_code_conversion_prompt(source_code)
        else:
            prompt = f"Analyze this code:\n\n```c\n{source_code}\n```"
        
        # Analyze source code prompt
        source_tokens = self.token_analyzer.count_tokens(prompt)
        
        # Compile source code
        compile_result = self.compiler.compile_c_code(source_code)
        
        if compile_result is None:
            return {
                'code_name': code_name,
                'prompt_type': prompt_type,
                'source_tokens': source_tokens,
                'binary_tokens': 0,
                'assembly_tokens': 0,
                'compilation_success': False,
                'binary_savings': 0,
                'assembly_savings': 0,
                'binary_savings_pct': 0,
                'assembly_savings_pct': 0,
                'estimated_cost_source': 0,
                'estimated_cost_binary': 0,
                'estimated_cost_assembly': 0
            }
        
        binary_hex, assembly_text = compile_result
        
        # Create machine code prompts
        binary_prompt = prompt.replace(f"```c\n{source_code}\n```", 
                                     f"```hex\n{binary_hex}\n```")
        assembly_prompt = prompt.replace(f"```c\n{source_code}\n```", 
                                       f"```assembly\n{assembly_text}\n```")
        
        # Count tokens for machine code prompts
        binary_tokens = self.token_analyzer.count_tokens(binary_prompt)
        assembly_tokens = self.token_analyzer.count_tokens(assembly_prompt)
        
        # Calculate savings
        binary_savings = source_tokens - binary_tokens
        assembly_savings = source_tokens - assembly_tokens
        
        binary_savings_pct = (binary_savings / source_tokens * 100) if source_tokens > 0 else 0
        assembly_savings_pct = (assembly_savings / source_tokens * 100) if source_tokens > 0 else 0
        
        # Estimate costs (assuming 500 output tokens on average)
        estimated_output_tokens = 500
        source_cost = self.cost_calculator.calculate_cost(
            source_tokens, estimated_output_tokens, self.config.model_name)
        binary_cost = self.cost_calculator.calculate_cost(
            binary_tokens, estimated_output_tokens, self.config.model_name)
        assembly_cost = self.cost_calculator.calculate_cost(
            assembly_tokens, estimated_output_tokens, self.config.model_name)
        
        return {
            'code_name': code_name,
            'prompt_type': prompt_type,
            'source_tokens': source_tokens,
            'binary_tokens': binary_tokens,
            'assembly_tokens': assembly_tokens,
            'compilation_success': True,
            'binary_savings': binary_savings,
            'assembly_savings': assembly_savings,
            'binary_savings_pct': binary_savings_pct,
            'assembly_savings_pct': assembly_savings_pct,
            'estimated_cost_source': source_cost,
            'estimated_cost_binary': binary_cost,
            'estimated_cost_assembly': assembly_cost
        }
    
    def run_comprehensive_analysis(self) -> pd.DataFrame:
        """Run comprehensive analysis across all samples and prompt types."""
        
        # Get all code samples
        basic_generator = ExtendedSampleGenerator()
        all_samples = basic_generator.get_all_extended_samples()
        
        # Add basic samples from original generator
        from llm_token_analysis import CodeSampleGenerator
        basic_samples = CodeSampleGenerator.get_sample_programs()
        all_samples.extend(basic_samples)
        
        prompt_types = ["explanation", "optimization", "debugging", "conversion"]
        
        print(f"Running comprehensive analysis on {len(all_samples)} code samples")
        print(f"across {len(prompt_types)} prompt types...")
        
        total_analyses = len(all_samples) * len(prompt_types)
        current_analysis = 0
        
        for code_name, source_code in all_samples:
            for prompt_type in prompt_types:
                current_analysis += 1
                print(f"Progress: {current_analysis}/{total_analyses} - {code_name} ({prompt_type})")
                
                result = self.analyze_prompt_scenario(code_name, source_code, prompt_type)
                self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive analysis report."""
        
        successful_df = df[df['compilation_success'] == True]
        
        if len(successful_df) == 0:
            return "No successful compilations to analyze."
        
        # Calculate aggregate statistics
        total_analyses = len(successful_df)
        avg_source_tokens = successful_df['source_tokens'].mean()
        avg_binary_tokens = successful_df['binary_tokens'].mean()
        avg_assembly_tokens = successful_df['assembly_tokens'].mean()
        
        avg_binary_savings = successful_df['binary_savings_pct'].mean()
        avg_assembly_savings = successful_df['assembly_savings_pct'].mean()
        
        # Cost analysis
        total_source_cost = successful_df['estimated_cost_source'].sum()
        total_binary_cost = successful_df['estimated_cost_binary'].sum()
        total_assembly_cost = successful_df['estimated_cost_assembly'].sum()
        
        binary_cost_savings = total_source_cost - total_binary_cost
        assembly_cost_savings = total_source_cost - total_assembly_cost
        
        # Best and worst cases
        best_binary_savings = successful_df['binary_savings_pct'].max()
        worst_binary_savings = successful_df['binary_savings_pct'].min()
        best_assembly_savings = successful_df['assembly_savings_pct'].max()
        worst_assembly_savings = successful_df['assembly_savings_pct'].min()
        
        # Analysis by prompt type
        prompt_analysis = successful_df.groupby('prompt_type').agg({
            'source_tokens': 'mean',
            'binary_savings_pct': 'mean',
            'assembly_savings_pct': 'mean',
            'estimated_cost_source': 'sum',
            'estimated_cost_binary': 'sum',
            'estimated_cost_assembly': 'sum'
        }).round(2)
        
        report = f"""
COMPREHENSIVE TOKEN COST ANALYSIS REPORT
========================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {self.config.model_name}
Total Analyses: {total_analyses}
Unique Code Samples: {len(successful_df['code_name'].unique())}
Prompt Types: {len(successful_df['prompt_type'].unique())}

OVERALL STATISTICS:
------------------
Average Token Counts:
- Source Code Prompts: {avg_source_tokens:.1f} tokens
- Binary Code Prompts: {avg_binary_tokens:.1f} tokens  
- Assembly Code Prompts: {avg_assembly_tokens:.1f} tokens

Token Savings Summary:
- Binary vs Source: {avg_binary_savings:.1f}% average savings
- Assembly vs Source: {avg_assembly_savings:.1f}% average savings

Best Case Scenarios:
- Binary Savings: {best_binary_savings:.1f}%
- Assembly Savings: {best_assembly_savings:.1f}%

Worst Case Scenarios:
- Binary Savings: {worst_binary_savings:.1f}%
- Assembly Savings: {worst_assembly_savings:.1f}%

COST ANALYSIS:
--------------
Total Estimated Costs (for all {total_analyses} analyses):
- Source Code Prompts: ${total_source_cost:.4f}
- Binary Code Prompts: ${total_binary_cost:.4f}
- Assembly Code Prompts: ${total_assembly_cost:.4f}

Potential Cost Savings:
- Binary vs Source: ${binary_cost_savings:.4f} ({binary_cost_savings/total_source_cost*100:.1f}%)
- Assembly vs Source: ${assembly_cost_savings:.4f} ({assembly_cost_savings/total_source_cost*100:.1f}%)

ANALYSIS BY PROMPT TYPE:
-----------------------
"""
        
        for prompt_type, stats in prompt_analysis.iterrows():
            cost_saved_binary = stats['estimated_cost_source'] - stats['estimated_cost_binary']
            cost_saved_assembly = stats['estimated_cost_source'] - stats['estimated_cost_assembly']
            
            report += f"""
{prompt_type.upper()}:
- Average Source Tokens: {stats['source_tokens']:.1f}
- Binary Savings: {stats['binary_savings_pct']:.1f}%
- Assembly Savings: {stats['assembly_savings_pct']:.1f}%
- Cost Savings (Binary): ${cost_saved_binary:.4f}
- Cost Savings (Assembly): ${cost_saved_assembly:.4f}
"""
        
        # Top performers
        top_binary_savers = successful_df.nlargest(5, 'binary_savings_pct')[
            ['code_name', 'prompt_type', 'binary_savings_pct']]
        top_assembly_savers = successful_df.nlargest(5, 'assembly_savings_pct')[
            ['code_name', 'prompt_type', 'assembly_savings_pct']]
        
        report += f"""
TOP TOKEN SAVERS:
----------------
Binary Code (Top 5):
{top_binary_savers.to_string(index=False)}

Assembly Code (Top 5):
{top_assembly_savers.to_string(index=False)}

IMPLICATIONS:
------------
1. Machine code representations can provide significant token savings
2. The effectiveness varies by code complexity and prompt type
3. Cost savings could be substantial for high-volume applications
4. Assembly code may be more human-readable than hex but less token-efficient

RECOMMENDATIONS:
---------------
1. Consider binary representation for simple, well-defined code tasks
2. Assembly may be better for debugging and optimization tasks
3. Cost-benefit analysis should include model performance on machine code
4. Further research needed on model accuracy with machine code inputs
"""
        
        return report
    
    def create_comprehensive_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations."""
        
        successful_df = df[df['compilation_success'] == True]
        
        if len(successful_df) == 0:
            print("No successful compilations to visualize.")
            return
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # Create a grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Token savings distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(successful_df['binary_savings_pct'], bins=20, alpha=0.7, 
                label='Binary', color='blue', edgecolor='black')
        ax1.hist(successful_df['assembly_savings_pct'], bins=20, alpha=0.7, 
                label='Assembly', color='orange', edgecolor='black')
        ax1.set_xlabel('Token Savings (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Token Savings')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Savings by prompt type
        ax2 = fig.add_subplot(gs[0, 1])
        prompt_stats = successful_df.groupby('prompt_type')[['binary_savings_pct', 'assembly_savings_pct']].mean()
        prompt_stats.plot(kind='bar', ax=ax2)
        ax2.set_xlabel('Prompt Type')
        ax2.set_ylabel('Average Savings (%)')
        ax2.set_title('Token Savings by Prompt Type')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cost savings analysis
        ax3 = fig.add_subplot(gs[0, 2])
        cost_data = successful_df.groupby('prompt_type')[['estimated_cost_source', 'estimated_cost_binary', 'estimated_cost_assembly']].sum()
        cost_data.plot(kind='bar', ax=ax3)
        ax3.set_xlabel('Prompt Type')
        ax3.set_ylabel('Total Cost ($)')
        ax3.set_title('Cost Analysis by Prompt Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Token count comparison
        ax4 = fig.add_subplot(gs[1, 0])
        token_means = successful_df.groupby('prompt_type')[['source_tokens', 'binary_tokens', 'assembly_tokens']].mean()
        token_means.plot(kind='bar', ax=ax4)
        ax4.set_xlabel('Prompt Type')
        ax4.set_ylabel('Average Token Count')
        ax4.set_title('Token Counts by Representation')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Scatter plot: Source tokens vs savings
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(successful_df['source_tokens'], successful_df['binary_savings_pct'], 
                   alpha=0.6, label='Binary', s=50)
        ax5.scatter(successful_df['source_tokens'], successful_df['assembly_savings_pct'], 
                   alpha=0.6, label='Assembly', s=50)
        ax5.set_xlabel('Source Token Count')
        ax5.set_ylabel('Savings (%)')
        ax5.set_title('Source Size vs Savings')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Heatmap of savings by code type and prompt type
        ax6 = fig.add_subplot(gs[1, 2])
        pivot_table = successful_df.pivot_table(
            values='binary_savings_pct', 
            index='code_name', 
            columns='prompt_type', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax6)
        ax6.set_title('Binary Savings Heatmap')
        ax6.set_xlabel('Prompt Type')
        ax6.set_ylabel('Code Sample')
        
        # 7. Cumulative savings potential
        ax7 = fig.add_subplot(gs[2, :])
        sorted_binary = successful_df.sort_values('binary_savings_pct', ascending=False)
        sorted_assembly = successful_df.sort_values('assembly_savings_pct', ascending=False)
        
        ax7.plot(range(len(sorted_binary)), sorted_binary['binary_savings_pct'].cumsum(), 
                label='Binary (Cumulative)', linewidth=2)
        ax7.plot(range(len(sorted_assembly)), sorted_assembly['assembly_savings_pct'].cumsum(), 
                label='Assembly (Cumulative)', linewidth=2)
        ax7.set_xlabel('Analysis Number (sorted by savings)')
        ax7.set_ylabel('Cumulative Savings (%)')
        ax7.set_title('Cumulative Token Savings Potential')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Token Cost Analysis Results', fontsize=16, y=0.98)
        plt.savefig('comprehensive_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, df: pd.DataFrame):
        """Save comprehensive results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        df.to_csv(f'comprehensive_analysis_{timestamp}.csv', index=False)
        
        # Save summary statistics
        summary_stats = df.groupby(['prompt_type', 'compilation_success']).agg({
            'source_tokens': ['mean', 'std', 'min', 'max'],
            'binary_savings_pct': ['mean', 'std', 'min', 'max'],
            'assembly_savings_pct': ['mean', 'std', 'min', 'max'],
            'estimated_cost_source': 'sum',
            'estimated_cost_binary': 'sum',
            'estimated_cost_assembly': 'sum'
        }).round(4)
        
        summary_stats.to_csv(f'analysis_summary_{timestamp}.csv')
        
        print(f"Results saved:")
        print(f"- Main results: comprehensive_analysis_{timestamp}.csv")
        print(f"- Summary stats: analysis_summary_{timestamp}.csv")
        print(f"- Visualizations: comprehensive_analysis_results.png")

def main():
    """Main function to run comprehensive analysis."""
    
    parser = argparse.ArgumentParser(description='Comprehensive LLM Token Cost Analysis')
    parser.add_argument('--model', default='gpt-4', help='Model to use for tokenization')
    parser.add_argument('--config', default='analysis_config.json', help='Configuration file')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.model_name = args.model
    config.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("COMPREHENSIVE LLM TOKEN COST ANALYSIS")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(config)
    
    # Run comprehensive analysis
    results_df = analyzer.run_comprehensive_analysis()
    
    # Generate report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_comprehensive_report(results_df)
    print(report)
    
    # Save report
    report_file = os.path.join(config.output_dir, 'comprehensive_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_comprehensive_visualizations(results_df)
    
    # Save results
    print("\nSaving results...")
    analyzer.save_results(results_df)
    
    print(f"\nAnalysis complete! Check the '{config.output_dir}' directory for all outputs.")

if __name__ == "__main__":
    main()