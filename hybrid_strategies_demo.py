#!/usr/bin/env python3
"""
Hybrid Context Strategies Demo
=============================

This demo shows how intelligent hybrid representations can provide
minimal but effective context for understanding machine code, without
requiring API calls.
"""

import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import List, Dict, Tuple

class HybridContextDemo:
    """Demonstrates hybrid context strategies with token analysis."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
    
    def extract_function_signature(self, source_code: str) -> str:
        """Extract minimal function signature information."""
        lines = source_code.split('\n')
        signatures = []
        
        for line in lines:
            line = line.strip()
            # Look for function definitions
            if (('(' in line and ')' in line and 
                 any(keyword in line for keyword in ['int ', 'void ', 'char ', 'float ', 'double '])) or
                'main(' in line):
                # Clean up the signature
                signature = line.replace('{', '').replace('    ', ' ').strip()
                if signature and len(signature) < 80:  # Keep it short
                    signatures.append(signature)
        
        return ' | '.join(signatures[:2])  # Max 2 signatures
    
    def extract_key_variables(self, source_code: str) -> str:
        """Extract key variable declarations and constants."""
        lines = source_code.split('\n')
        variables = []
        
        for line in lines:
            line = line.strip()
            # Look for variable declarations with initialization
            if (any(keyword in line for keyword in ['int ', 'char ', 'float ']) and 
                '=' in line and ';' in line and len(line) < 60):
                # Clean up variable declaration
                var_decl = line.replace(';', '').replace('    ', ' ').strip()
                variables.append(var_decl)
        
        return ' | '.join(variables[:3])  # Max 3 variable declarations
    
    def analyze_control_flow(self, source_code: str) -> str:
        """Analyze control flow patterns in source code."""
        patterns = []
        
        if 'for(' in source_code or 'for ' in source_code:
            patterns.append('LOOP')
        if 'while(' in source_code or 'while ' in source_code:
            patterns.append('WHILE')
        if 'if(' in source_code or 'if ' in source_code:
            patterns.append('BRANCH')
        if 'switch(' in source_code:
            patterns.append('SWITCH')
        if 'return' in source_code:
            patterns.append('RETURN')
        if 'printf' in source_code or 'print' in source_code:
            patterns.append('OUTPUT')
        
        return '>'.join(patterns[:4])  # Max 4 patterns
    
    def compress_opcodes_intelligently(self, opcodes: str) -> str:
        """Compress opcodes using pattern recognition."""
        if len(opcodes) <= 32:
            return opcodes
        
        # Look for repeating 8-character patterns (ARM64 instructions)
        patterns = {}
        for i in range(0, len(opcodes) - 7, 8):
            pattern = opcodes[i:i+8]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # If we have significant repetition, create a compressed version
        if patterns and max(patterns.values()) > 1:
            # Keep first occurrence and note repetitions
            common_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
            result = opcodes[:24]  # First 3 instructions
            for pattern, count in common_patterns[:2]:
                if count > 1:
                    result += f"*{pattern}x{count}"
            return result[:64]  # Limit total length
        
        # Otherwise, just truncate intelligently (keep beginning and end)
        if len(opcodes) > 48:
            return opcodes[:32] + ".." + opcodes[-16:]
        
        return opcodes[:48]
    
    def generate_hybrid_representations(self, source_code: str, opcodes: str) -> Dict[str, str]:
        """Generate various hybrid representations."""
        
        signatures = self.extract_function_signature(source_code)
        variables = self.extract_key_variables(source_code)
        control_flow = self.analyze_control_flow(source_code)
        compressed_opcodes = self.compress_opcodes_intelligently(opcodes)
        
        representations = {
            # Pure baselines
            'pure_source': source_code,
            'pure_opcodes': opcodes,
            
            # Minimal hybrid approaches
            'minimal_context': f"fn:{signatures.split('|')[0] if signatures else 'main'} | {compressed_opcodes}",
            
            # Function + control flow
            'function_flow': f"sig:{signatures} | cf:{control_flow} | {compressed_opcodes}",
            
            # Variables + opcodes
            'vars_opcodes': f"vars:{variables} | {compressed_opcodes}",
            
            # Complete minimal context
            'complete_context': f"fn:{signatures} | vars:{variables} | cf:{control_flow} | opc:{compressed_opcodes}",
            
            # Smart adaptive
            'adaptive': self._generate_adaptive_representation(
                signatures, variables, control_flow, compressed_opcodes
            ),
            
            # Compressed source + opcodes
            'compressed_hybrid': f"{self._compress_source(source_code)} | {compressed_opcodes}",
        }
        
        return representations
    
    def _generate_adaptive_representation(self, signatures: str, variables: str, 
                                        control_flow: str, opcodes: str) -> str:
        """Generate adaptive representation based on content analysis."""
        
        components = []
        
        # Always include function signature if available and short
        if signatures and len(signatures) < 50:
            components.append(f"fn:{signatures.split('|')[0]}")
        
        # Include control flow if complex (multiple patterns)
        if len(control_flow.split('>')) > 1:
            components.append(f"cf:{control_flow}")
        
        # Include variables if they provide value and aren't too long
        if variables and len(variables) < 50:
            components.append(f"vars:{variables.split('|')[0]}")
        
        # Always include compressed opcodes
        components.append(opcodes)
        
        return ' | '.join(components)
    
    def _compress_source(self, source_code: str) -> str:
        """Create a compressed version of source code."""
        # Remove comments, extra whitespace, and keep only essential parts
        lines = [line.strip() for line in source_code.split('\n') if line.strip()]
        essential_lines = []
        
        for line in lines:
            # Skip includes and braces
            if line.startswith('#') or line in ['{', '}']:
                continue
            # Keep function signatures and key statements
            if (any(keyword in line for keyword in ['int ', 'void ', 'char ', 'main']) or
                any(keyword in line for keyword in ['for', 'while', 'if', 'return', 'printf']) or
                '=' in line):
                essential_lines.append(line)
        
        compressed = '; '.join(essential_lines[:4])  # Max 4 lines
        return compressed[:100] + '...' if len(compressed) > 100 else compressed
    
    def analyze_token_efficiency(self, representations: Dict[str, str]) -> pd.DataFrame:
        """Analyze token efficiency of different representations."""
        
        results = []
        source_tokens = len(self.tokenizer.encode(representations['pure_source']))
        
        for repr_name, repr_content in representations.items():
            tokens = len(self.tokenizer.encode(repr_content))
            efficiency = ((source_tokens - tokens) / source_tokens * 100) if source_tokens > 0 else 0
            
            # Calculate information density
            char_to_token_ratio = len(repr_content) / tokens if tokens > 0 else 0
            
            # Estimate context quality (heuristic)
            context_quality = self._estimate_context_quality(repr_name, repr_content)
            
            results.append({
                'representation': repr_name,
                'content': repr_content[:100] + '...' if len(repr_content) > 100 else repr_content,
                'characters': len(repr_content),
                'tokens': tokens,
                'token_efficiency_pct': efficiency,
                'char_per_token': char_to_token_ratio,
                'context_quality': context_quality,
                'efficiency_score': efficiency * context_quality / 100  # Combined metric
            })
        
        return pd.DataFrame(results)
    
    def _estimate_context_quality(self, repr_name: str, content: str) -> float:
        """Estimate context quality using heuristics."""
        
        if repr_name == 'pure_source':
            return 1.0  # Perfect context
        elif repr_name == 'pure_opcodes':
            return 0.3  # Minimal context
        
        quality_score = 0.5  # Base score
        
        # Bonus for including function signatures
        if 'fn:' in content or 'sig:' in content:
            quality_score += 0.2
        
        # Bonus for control flow information
        if 'cf:' in content:
            quality_score += 0.15
        
        # Bonus for variable information
        if 'vars:' in content:
            quality_score += 0.1
        
        # Bonus for balanced length (not too short, not too long)
        if 50 <= len(content) <= 200:
            quality_score += 0.1
        elif len(content) < 20:
            quality_score -= 0.2
        
        return min(1.0, quality_score)
    
    def create_demo_visualizations(self, results_df: pd.DataFrame, test_case_name: str):
        """Create visualizations for the demo."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Token count comparison
        ax1 = axes[0, 0]
        
        # Sort by tokens for better visualization
        sorted_df = results_df.sort_values('tokens')
        
        colors = ['red' if 'pure_source' in repr else 'orange' if 'pure_opcodes' in repr 
                 else 'green' for repr in sorted_df['representation']]
        
        bars = ax1.barh(range(len(sorted_df)), sorted_df['tokens'], color=colors, alpha=0.8)
        ax1.set_title(f'Token Count by Representation - {test_case_name}')
        ax1.set_xlabel('Number of Tokens')
        ax1.set_yticks(range(len(sorted_df)))
        ax1.set_yticklabels(sorted_df['representation'])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_df['tokens'])):
            ax1.text(value + 2, bar.get_y() + bar.get_height()/2,
                    f'{value}', va='center', fontweight='bold')
        
        # 2. Efficiency vs Quality trade-off
        ax2 = axes[0, 1]
        
        scatter = ax2.scatter(results_df['token_efficiency_pct'], results_df['context_quality'], 
                             s=100, alpha=0.7, c=range(len(results_df)), cmap='viridis')
        
        # Add labels to points
        for i, row in results_df.iterrows():
            ax2.annotate(row['representation'], 
                        (row['token_efficiency_pct'], row['context_quality']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Token Efficiency (%)')
        ax2.set_ylabel('Context Quality (0-1)')
        ax2.set_title('Efficiency vs Quality Trade-off')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Quality Threshold')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Efficiency Baseline')
        ax2.legend()
        
        # 3. Combined efficiency score
        ax3 = axes[1, 0]
        
        efficiency_sorted = results_df.sort_values('efficiency_score', ascending=True)
        colors = ['lightcoral' if score < 0 else 'lightgreen' if score > 20 else 'khaki' 
                 for score in efficiency_sorted['efficiency_score']]
        
        bars = ax3.barh(range(len(efficiency_sorted)), efficiency_sorted['efficiency_score'], 
                       color=colors, alpha=0.8)
        ax3.set_title('Combined Efficiency Score (Efficiency × Quality)')
        ax3.set_xlabel('Efficiency Score')
        ax3.set_yticks(range(len(efficiency_sorted)))
        ax3.set_yticklabels(efficiency_sorted['representation'])
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, efficiency_sorted['efficiency_score'])):
            ax3.text(value + (1 if value > 0 else -1), bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}', va='center', ha='left' if value > 0 else 'right', fontweight='bold')
        
        # 4. Character efficiency
        ax4 = axes[1, 1]
        
        ax4.scatter(results_df['characters'], results_df['tokens'], s=100, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(results_df['characters'], results_df['tokens'], 1)
        p = np.poly1d(z)
        ax4.plot(results_df['characters'], p(results_df['characters']), "r--", alpha=0.8)
        
        # Add labels
        for i, row in results_df.iterrows():
            ax4.annotate(row['representation'], 
                        (row['characters'], row['tokens']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Character Count')
        ax4.set_ylabel('Token Count')
        ax4.set_title('Character to Token Conversion Efficiency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'hybrid_context_demo_{test_case_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_analysis_report(self, results_df: pd.DataFrame, test_case_name: str):
        """Print detailed analysis report."""
        
        print(f"\n{'='*60}")
        print(f"HYBRID CONTEXT ANALYSIS: {test_case_name.upper()}")
        print(f"{'='*60}")
        
        # Summary statistics
        source_tokens = results_df[results_df['representation'] == 'pure_source']['tokens'].iloc[0]
        opcodes_tokens = results_df[results_df['representation'] == 'pure_opcodes']['tokens'].iloc[0]
        baseline_efficiency = ((source_tokens - opcodes_tokens) / source_tokens * 100)
        
        print(f"📊 BASELINE COMPARISON:")
        print(f"   Pure Source: {source_tokens} tokens")
        print(f"   Pure Opcodes: {opcodes_tokens} tokens ({baseline_efficiency:+.1f}% efficiency)")
        
        # Best hybrid approaches
        hybrid_results = results_df[~results_df['representation'].isin(['pure_source', 'pure_opcodes'])]
        best_efficiency = hybrid_results.loc[hybrid_results['token_efficiency_pct'].idxmax()]
        best_quality = hybrid_results.loc[hybrid_results['context_quality'].idxmax()]
        best_combined = hybrid_results.loc[hybrid_results['efficiency_score'].idxmax()]
        
        print(f"\n🏆 BEST HYBRID APPROACHES:")
        print(f"   Best Efficiency: {best_efficiency['representation']} ({best_efficiency['token_efficiency_pct']:+.1f}%)")
        print(f"   Best Quality: {best_quality['representation']} (quality: {best_quality['context_quality']:.2f})")
        print(f"   Best Combined: {best_combined['representation']} (score: {best_combined['efficiency_score']:.1f})")
        
        # Detailed breakdown
        print(f"\n📋 DETAILED BREAKDOWN:")
        for _, row in results_df.iterrows():
            print(f"\n   {row['representation'].upper()}:")
            print(f"     Tokens: {row['tokens']} ({row['token_efficiency_pct']:+.1f}% vs source)")
            print(f"     Quality: {row['context_quality']:.2f}")
            print(f"     Content: {row['content']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if best_combined['efficiency_score'] > 15:
            print(f"   ✅ RECOMMENDED: {best_combined['representation']} provides excellent balance")
        elif best_combined['efficiency_score'] > 5:
            print(f"   ⚠️  CONDITIONAL: {best_combined['representation']} shows promise")
        else:
            print(f"   ❌ NEEDS WORK: All hybrid approaches need improvement")
        
        if baseline_efficiency > 50:
            print(f"   💰 COST SAVINGS: Pure opcodes already provide {baseline_efficiency:.0f}% token reduction")
        
        print(f"   🎯 OPTIMAL STRATEGY: Use {best_combined['representation']} for balance of efficiency and context")

def get_demo_test_cases() -> List[Tuple[str, str, str]]:
    """Get test cases for the demo."""
    
    return [
        ("Hello World", 
         """#include <stdio.h>
int main() {
    printf("Hello, World!\\n");
    return 0;
}""",
         "a9bf7bfd910003fd90000000913e800094000004528000008c17bfdd65f03c0"),
        
        ("Factorial Recursive",
         """#include <stdio.h>
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
int main() {
    int result = factorial(5);
    printf("5! = %d\\n", result);
    return 0;
}""",
         "7100043f5400004c528000201b007c20d65f03c052800140aa0003e094000001b900037f90000000913e8000940000045280000052800000d65f03c0"),
        
        ("Array Processing",
         """#include <stdio.h>
int main() {
    int numbers[] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += numbers[i];
    }
    printf("Sum: %d\\n", sum);
    return 0;
}""",
         "52800020528000409900037f528000809900037f528000a09900037f528000c09900037f528000e09900037f528000009900037f7100014054000060b8606800"),
        
        ("String Length",
         """#include <stdio.h>
int main() {
    char str[] = "hello";
    int length = 0;
    while (str[length] != '\\0') {
        length++;
    }
    printf("Length: %d\\n", length);
    return 0;
}""",
         "528d4ca0b900037f90000000f90007e0528000009900037f39400000710001405400004811000400390000007100014054ffffc0")
    ]

def main():
    """Run the hybrid context demo."""
    
    print("🔬 HYBRID CONTEXT STRATEGIES DEMO")
    print("=" * 50)
    print("Testing intelligent hybrid representations that combine")
    print("minimal source context with machine code opcodes.\n")
    
    # Initialize demo
    demo = HybridContextDemo("gpt-4")
    
    # Get test cases
    test_cases = get_demo_test_cases()
    
    print(f"📋 Test Cases: {len(test_cases)}")
    for name, _, _ in test_cases:
        print(f"  • {name}")
    
    # Process each test case
    all_results = []
    
    for name, source_code, opcodes in test_cases:
        print(f"\n{'='*40}")
        print(f"🧪 ANALYZING: {name}")
        print(f"{'='*40}")
        
        # Generate hybrid representations
        representations = demo.generate_hybrid_representations(source_code, opcodes)
        
        print(f"\n🔧 Generated {len(representations)} representations:")
        for repr_name in representations.keys():
            print(f"  • {repr_name}")
        
        # Analyze token efficiency
        results_df = demo.analyze_token_efficiency(representations)
        
        # Create visualizations
        demo.create_demo_visualizations(results_df, name)
        
        # Print analysis
        demo.print_analysis_report(results_df, name)
        
        # Save results
        results_df['test_case'] = name
        all_results.append(results_df)
        
        print(f"\n📁 Visualization saved as: hybrid_context_demo_{name.lower().replace(' ', '_')}.png")
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv('hybrid_context_demo_results.csv', index=False)
    
    # Overall analysis
    print(f"\n{'='*60}")
    print(f"🎯 OVERALL ANALYSIS ACROSS ALL TEST CASES")
    print(f"{'='*60}")
    
    # Calculate average metrics by representation
    avg_metrics = combined_results.groupby('representation').agg({
        'token_efficiency_pct': 'mean',
        'context_quality': 'mean',
        'efficiency_score': 'mean',
        'tokens': 'mean'
    }).round(2)
    
    print(f"\n📊 AVERAGE PERFORMANCE BY REPRESENTATION:")
    for repr_name, metrics in avg_metrics.iterrows():
        print(f"\n{repr_name.upper()}:")
        print(f"  Token Efficiency: {metrics['token_efficiency_pct']:+.1f}%")
        print(f"  Context Quality: {metrics['context_quality']:.2f}")
        print(f"  Combined Score: {metrics['efficiency_score']:.1f}")
        print(f"  Avg Tokens: {metrics['tokens']:.0f}")
    
    # Find best overall approach
    best_overall = avg_metrics.loc[avg_metrics['efficiency_score'].idxmax()]
    print(f"\n🏆 BEST OVERALL APPROACH: {avg_metrics['efficiency_score'].idxmax()}")
    print(f"   Average efficiency: {best_overall['token_efficiency_pct']:+.1f}%")
    print(f"   Average quality: {best_overall['context_quality']:.2f}")
    print(f"   Combined score: {best_overall['efficiency_score']:.1f}")
    
    # Implementation recommendations
    print(f"\n💡 IMPLEMENTATION RECOMMENDATIONS:")
    
    # Analyze which approaches work best
    hybrid_only = avg_metrics.drop(['pure_source', 'pure_opcodes'])
    
    if hybrid_only['efficiency_score'].max() > 20:
        best_hybrid = hybrid_only['efficiency_score'].idxmax()
        print(f"   ✅ DEPLOY HYBRID: {best_hybrid} shows strong performance")
        print(f"      - Provides {avg_metrics.loc[best_hybrid, 'token_efficiency_pct']:+.1f}% token savings")
        print(f"      - Maintains {avg_metrics.loc[best_hybrid, 'context_quality']:.0%} context quality")
    elif hybrid_only['efficiency_score'].max() > 10:
        print(f"   ⚠️  HYBRID PROMISING: Further optimization recommended")
        print(f"      - Best hybrid: {hybrid_only['efficiency_score'].idxmax()}")
        print(f"      - Focus on improving context extraction")
    else:
        print(f"   🔴 USE PURE OPCODES: Hybrid approaches add overhead")
        opcodes_eff = avg_metrics.loc['pure_opcodes', 'token_efficiency_pct']
        print(f"      - Pure opcodes provide {opcodes_eff:+.1f}% efficiency")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Test best approach with real LLM APIs")
    print(f"   2. Validate quality with human evaluation")
    print(f"   3. Optimize context extraction algorithms")
    print(f"   4. Scale testing to larger code samples")
    
    print(f"\n📁 FILES GENERATED:")
    print(f"   • hybrid_context_demo_results.csv - Complete results")
    print(f"   • hybrid_context_demo_*.png - Visualizations for each test case")
    
    print(f"\n✅ Demo complete! Ready for real LLM validation.")

if __name__ == "__main__":
    main()