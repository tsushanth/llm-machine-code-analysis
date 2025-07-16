#!/usr/bin/env python3
"""
Advanced Optimization Research Framework
=======================================

Explores advanced techniques to further improve token efficiency beyond
our baseline 70% savings, including hybrid representations, compression,
and model-specific optimizations.
"""

import os
import subprocess
import tempfile
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import zlib
import base64
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class OptimizationStrategy:
    """Represents an optimization strategy for machine code representation."""
    name: str
    description: str
    implementation: callable
    category: str  # 'compression', 'hybrid', 'encoding', 'semantic'

class AdvancedOptimizer:
    """Framework for testing advanced optimization strategies."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
        self.temp_dir = tempfile.mkdtemp()
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> List[OptimizationStrategy]:
        """Initialize all optimization strategies."""
        
        return [
            # Compression strategies
            OptimizationStrategy(
                "hex_compression", 
                "Compress hex opcodes using zlib",
                self._compress_hex_opcodes,
                "compression"
            ),
            OptimizationStrategy(
                "run_length_encoding",
                "Apply run-length encoding to repetitive patterns",
                self._run_length_encode,
                "compression"
            ),
            OptimizationStrategy(
                "dictionary_compression",
                "Use dictionary compression for common instruction patterns",
                self._dictionary_compress,
                "compression"
            ),
            
            # Encoding strategies
            OptimizationStrategy(
                "base64_encoding",
                "Use base64 encoding instead of hex",
                self._base64_encode,
                "encoding"
            ),
            OptimizationStrategy(
                "instruction_abbreviation",
                "Use abbreviated instruction mnemonics",
                self._abbreviate_instructions,
                "encoding"
            ),
            OptimizationStrategy(
                "token_aware_formatting",
                "Format opcodes to align with tokenizer boundaries",
                self._token_aware_format,
                "encoding"
            ),
            
            # Hybrid strategies
            OptimizationStrategy(
                "source_plus_opcodes",
                "Combine source comments with opcodes",
                self._hybrid_source_opcodes,
                "hybrid"
            ),
            OptimizationStrategy(
                "selective_conversion",
                "Convert only performance-critical sections to opcodes",
                self._selective_conversion,
                "hybrid"
            ),
            OptimizationStrategy(
                "hierarchical_representation",
                "Multi-level representation with different detail levels",
                self._hierarchical_representation,
                "hybrid"
            ),
            
            # Semantic strategies
            OptimizationStrategy(
                "semantic_chunking",
                "Group instructions by semantic meaning",
                self._semantic_chunking,
                "semantic"
            ),
            OptimizationStrategy(
                "control_flow_annotation",
                "Add control flow metadata to opcodes",
                self._control_flow_annotation,
                "semantic"
            ),
            OptimizationStrategy(
                "register_abstraction",
                "Abstract register usage patterns",
                self._register_abstraction,
                "semantic"
            )
        ]
    
    # Compression Strategy Implementations
    def _compress_hex_opcodes(self, opcodes: str, source: str, assembly: str) -> str:
        """Compress hex opcodes using zlib."""
        if not opcodes:
            return ""
        
        # Compress the hex string
        compressed = zlib.compress(opcodes.encode('utf-8'))
        # Convert to base64 for text representation
        compressed_b64 = base64.b64encode(compressed).decode('utf-8')
        
        # Add header to indicate compression
        return f"COMPRESSED:{compressed_b64}"
    
    def _run_length_encode(self, opcodes: str, source: str, assembly: str) -> str:
        """Apply run-length encoding to repetitive patterns."""
        if not opcodes:
            return ""
        
        # Find repeating 8-character patterns (ARM64 instructions)
        result = []
        i = 0
        while i < len(opcodes):
            if i + 8 <= len(opcodes):
                pattern = opcodes[i:i+8]
                count = 1
                
                # Count repetitions
                j = i + 8
                while j + 8 <= len(opcodes) and opcodes[j:j+8] == pattern:
                    count += 1
                    j += 8
                
                # Encode if repetition found
                if count > 1:
                    result.append(f"{pattern}*{count}")
                    i = j
                else:
                    result.append(pattern)
                    i += 8
            else:
                result.append(opcodes[i:])
                break
        
        return "RLE:" + "|".join(result)
    
    def _dictionary_compress(self, opcodes: str, source: str, assembly: str) -> str:
        """Use dictionary compression for common ARM64 instruction patterns."""
        if not opcodes:
            return ""
        
        # Common ARM64 instruction patterns (first 4 bits often indicate instruction type)
        common_patterns = {
            "5280": "MOV_IMM",    # mov with immediate
            "a9bf": "STP_PRE",    # stp with pre-index
            "d65f": "RET",        # return instruction
            "9000": "ADRP",       # address page
            "9100": "ADD_IMM",    # add immediate
            "b900": "STR_IMM",    # store immediate
            "f900": "STR_64",     # 64-bit store
            "9400": "BL",         # branch with link
        }
        
        result = opcodes
        dictionary = {}
        
        # Replace common patterns
        for pattern, abbrev in common_patterns.items():
            if pattern in result:
                result = result.replace(pattern, abbrev)
                dictionary[abbrev] = pattern
        
        # Return compressed version with dictionary
        if dictionary:
            dict_str = json.dumps(dictionary)
            return f"DICT:{dict_str}|{result}"
        else:
            return opcodes
    
    # Encoding Strategy Implementations
    def _base64_encode(self, opcodes: str, source: str, assembly: str) -> str:
        """Convert hex opcodes to base64 encoding."""
        if not opcodes:
            return ""
        
        try:
            # Convert hex to bytes then to base64
            hex_bytes = bytes.fromhex(opcodes)
            b64_encoded = base64.b64encode(hex_bytes).decode('utf-8')
            return f"B64:{b64_encoded}"
        except ValueError:
            return opcodes  # Fallback to original if hex is invalid
    
    def _abbreviate_instructions(self, opcodes: str, source: str, assembly: str) -> str:
        """Create abbreviated assembly instructions."""
        if not assembly:
            return opcodes
        
        # Instruction abbreviations
        abbreviations = {
            'stp': 'S', 'ldp': 'L', 'mov': 'M', 'add': 'A', 'sub': 'U',
            'str': 'W', 'ldr': 'R', 'ret': 'T', 'bl': 'C', 'b.': 'J',
            'cmp': 'P', 'adrp': 'G', 'x29': '29', 'x30': '30', 'sp': 'SP',
            '#0x': '#', '[sp': '[S', ', #': ',#'
        }
        
        lines = assembly.split('\n')
        abbreviated_lines = []
        
        for line in lines:
            abbreviated = line
            for full, abbrev in abbreviations.items():
                abbreviated = abbreviated.replace(full, abbrev)
            abbreviated_lines.append(abbreviated)
        
        return "ABV:" + "|".join(abbreviated_lines)
    
    def _token_aware_format(self, opcodes: str, source: str, assembly: str) -> str:
        """Format opcodes to align with tokenizer boundaries."""
        if not opcodes:
            return ""
        
        # Analyze how the tokenizer splits hex strings
        test_strings = [opcodes[i:i+4] for i in range(0, min(len(opcodes), 32), 4)]
        
        # Find optimal grouping size
        best_grouping = 4
        best_efficiency = 0
        
        for group_size in [2, 3, 4, 6, 8]:
            grouped = " ".join([opcodes[i:i+group_size] for i in range(0, len(opcodes), group_size)])
            tokens = len(self.tokenizer.encode(grouped))
            efficiency = len(opcodes) / tokens
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_grouping = group_size
        
        # Apply optimal grouping
        optimized = " ".join([opcodes[i:i+best_grouping] for i in range(0, len(opcodes), best_grouping)])
        return f"TOK:{optimized}"
    
    # Hybrid Strategy Implementations
    def _hybrid_source_opcodes(self, opcodes: str, source: str, assembly: str) -> str:
        """Combine source comments with opcodes."""
        if not opcodes or not source:
            return opcodes
        
        # Extract key information from source
        source_lines = source.split('\n')
        key_lines = []
        
        for line in source_lines:
            stripped = line.strip()
            # Keep function definitions, important comments, and variable declarations
            if (any(keyword in stripped for keyword in ['int ', 'char ', 'float ', 'double ']) or
                stripped.startswith('//') or 
                stripped.startswith('/*') or
                '(' in stripped and ')' in stripped):
                key_lines.append(stripped)
        
        # Combine with opcodes
        context = " ".join(key_lines[:3])  # Limit context
        return f"CTX:{context}|OPC:{opcodes}"
    
    def _selective_conversion(self, opcodes: str, source: str, assembly: str) -> str:
        """Convert only performance-critical sections to opcodes."""
        if not source:
            return opcodes
        
        # Identify performance-critical patterns in source
        critical_patterns = ['for ', 'while ', '*', '++', '--', '[]']
        
        source_lines = source.split('\n')
        critical_lines = []
        
        for i, line in enumerate(source_lines):
            if any(pattern in line for pattern in critical_patterns):
                critical_lines.append(f"L{i}:{line.strip()}")
        
        # For critical sections, use opcodes; for others, use source
        if critical_lines:
            critical_context = "|".join(critical_lines[:5])  # Limit to 5 lines
            return f"CRIT:{critical_context}|OPC:{opcodes[:64]}"  # Limit opcodes too
        else:
            return source[:200]  # Use source for non-critical code
    
    def _hierarchical_representation(self, opcodes: str, source: str, assembly: str) -> str:
        """Create multi-level representation."""
        if not opcodes:
            return ""
        
        # Level 1: Function signature (from source)
        func_signature = ""
        if source:
            for line in source.split('\n'):
                if '(' in line and ')' in line and any(t in line for t in ['int ', 'void ', 'char ']):
                    func_signature = line.strip()
                    break
        
        # Level 2: Control flow summary
        control_flow = "LINEAR"
        if source:
            if 'for' in source or 'while' in source:
                control_flow = "LOOP"
            if 'if' in source:
                control_flow = "BRANCH"
        
        # Level 3: Compressed opcodes (first 32 chars)
        compressed_opcodes = opcodes[:32] if len(opcodes) > 32 else opcodes
        
        return f"SIG:{func_signature}|CF:{control_flow}|OPC:{compressed_opcodes}"
    
    # Semantic Strategy Implementations
    def _semantic_chunking(self, opcodes: str, source: str, assembly: str) -> str:
        """Group instructions by semantic meaning."""
        if not assembly:
            return opcodes
        
        # Categorize instructions by type
        categories = {
            'SETUP': ['stp', 'mov x29', 'sub sp'],
            'LOAD': ['ldr', 'ldp'],
            'STORE': ['str', 'stp'],
            'CALC': ['add', 'sub', 'mul', 'div'],
            'BRANCH': ['b.', 'bl', 'ret'],
            'CMP': ['cmp', 'tst']
        }
        
        lines = assembly.split('\n')
        chunks = []
        
        for line in lines:
            categorized = False
            for category, instructions in categories.items():
                if any(instr in line.lower() for instr in instructions):
                    chunks.append(f"{category}:{len(chunks)}")
                    categorized = True
                    break
            if not categorized:
                chunks.append(f"OTHER:{len(chunks)}")
        
        return "SEM:" + "|".join(chunks[:20])  # Limit chunks
    
    def _control_flow_annotation(self, opcodes: str, source: str, assembly: str) -> str:
        """Add control flow metadata to opcodes."""
        if not opcodes:
            return ""
        
        # Analyze control flow patterns
        cf_patterns = []
        
        if assembly:
            lines = assembly.split('\n')
            for line in lines:
                if 'bl ' in line:  # Function call
                    cf_patterns.append('CALL')
                elif 'b.' in line:  # Conditional branch
                    cf_patterns.append('COND')
                elif 'ret' in line:  # Return
                    cf_patterns.append('RET')
        
        # Combine with opcodes
        cf_summary = ">".join(cf_patterns[:10])  # Limit patterns
        return f"CF:{cf_summary}|{opcodes[:48]}"
    
    def _register_abstraction(self, opcodes: str, source: str, assembly: str) -> str:
        """Abstract register usage patterns."""
        if not assembly:
            return opcodes
        
        # Track register usage patterns
        register_map = {}
        register_counter = 0
        
        # Common ARM64 registers to abstract
        registers = ['x0', 'x1', 'x2', 'x8', 'x29', 'x30', 'w0', 'w1', 'w8']
        
        for reg in registers:
            if reg in assembly:
                register_map[reg] = f"R{register_counter}"
                register_counter += 1
        
        # Create abstracted representation
        abstracted = assembly
        for orig_reg, abstract_reg in register_map.items():
            abstracted = abstracted.replace(orig_reg, abstract_reg)
        
        # Compress abstracted assembly
        abstracted_lines = abstracted.split('\n')[:5]  # Limit lines
        return "REG:" + "|".join([line.strip() for line in abstracted_lines if line.strip()])
    
    def apply_optimization_strategy(self, strategy: OptimizationStrategy, 
                                  opcodes: str, source: str, assembly: str) -> str:
        """Apply a specific optimization strategy."""
        try:
            return strategy.implementation(opcodes, source, assembly)
        except Exception as e:
            print(f"Error applying {strategy.name}: {e}")
            return opcodes  # Fallback to original
    
    def run_optimization_analysis(self, test_programs: List[Tuple[str, str, str, str]]) -> pd.DataFrame:
        """Run comprehensive optimization analysis."""
        
        results = []
        
        print(f"Testing {len(self.strategies)} optimization strategies on {len(test_programs)} programs")
        
        for prog_name, source, opcodes, assembly in test_programs:
            print(f"\nAnalyzing {prog_name}...")
            
            # Baseline measurements
            source_tokens = len(self.tokenizer.encode(source))
            opcode_tokens = len(self.tokenizer.encode(opcodes))
            baseline_efficiency = (source_tokens - opcode_tokens) / source_tokens * 100
            
            print(f"  Baseline: {source_tokens} → {opcode_tokens} tokens ({baseline_efficiency:.1f}% efficiency)")
            
            # Test each optimization strategy
            for strategy in self.strategies:
                optimized_repr = self.apply_optimization_strategy(strategy, opcodes, source, assembly)
                
                if optimized_repr:
                    optimized_tokens = len(self.tokenizer.encode(optimized_repr))
                    optimized_efficiency = (source_tokens - optimized_tokens) / source_tokens * 100
                    improvement = optimized_efficiency - baseline_efficiency
                    
                    results.append({
                        'program': prog_name,
                        'strategy': strategy.name,
                        'category': strategy.category,
                        'source_tokens': source_tokens,
                        'baseline_tokens': opcode_tokens,
                        'optimized_tokens': optimized_tokens,
                        'baseline_efficiency': baseline_efficiency,
                        'optimized_efficiency': optimized_efficiency,
                        'improvement': improvement,
                        'compression_ratio': len(optimized_repr) / len(opcodes) if opcodes else 1.0,
                        'representation_length': len(optimized_repr)
                    })
                    
                    print(f"    {strategy.name}: {optimized_tokens} tokens ({optimized_efficiency:.1f}%, {improvement:+.1f}%)")
        
        return pd.DataFrame(results)
    
    def create_optimization_visualizations(self, df: pd.DataFrame):
        """Create comprehensive optimization analysis visualizations."""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Strategy performance comparison
        ax1 = axes[0, 0]
        strategy_performance = df.groupby('strategy')['improvement'].mean().sort_values(ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in strategy_performance.values]
        bars = ax1.bar(range(len(strategy_performance)), strategy_performance.values, color=colors, alpha=0.8)
        ax1.set_title('Average Improvement by Optimization Strategy')
        ax1.set_ylabel('Efficiency Improvement (%)')
        ax1.set_xticks(range(len(strategy_performance)))
        ax1.set_xticklabels(strategy_performance.index, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, strategy_performance.values)):
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    value + (0.5 if value > 0 else -0.5),
                    f'{value:+.1f}%', ha='center', 
                    va='bottom' if value > 0 else 'top', fontweight='bold', fontsize=9)
        
        # 2. Category performance
        ax2 = axes[0, 1]
        category_performance = df.groupby('category')['improvement'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        ax2.bar(category_performance.index, category_performance['mean'], 
               yerr=category_performance['std'], capsize=5, alpha=0.8)
        ax2.set_title('Average Improvement by Strategy Category')
        ax2.set_ylabel('Efficiency Improvement (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # 3. Compression ratio vs efficiency improvement
        ax3 = axes[1, 0]
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            ax3.scatter(cat_data['compression_ratio'], cat_data['improvement'], 
                       label=category, alpha=0.7, s=80)
        
        ax3.set_xlabel('Compression Ratio (optimized/original length)')
        ax3.set_ylabel('Efficiency Improvement (%)')
        ax3.set_title('Compression Ratio vs Efficiency Improvement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        
        # 4. Token reduction distribution
        ax4 = axes[1, 1]
        
        # Create bins for improvement ranges
        bins = [-20, -10, -5, 0, 5, 10, 20, 50]
        labels = ['<-10%', '-10 to -5%', '-5 to 0%', '0 to 5%', '5 to 10%', '10 to 20%', '>20%']
        
        ax4.hist(df['improvement'], bins=bins, edgecolor='black', alpha=0.8)
        ax4.set_xlabel('Efficiency Improvement (%)')
        ax4.set_ylabel('Number of Strategy-Program Combinations')
        ax4.set_title('Distribution of Optimization Results')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.grid(True, alpha=0.3)
        
        # 5. Program-specific analysis
        ax5 = axes[2, 0]
        program_performance = df.groupby('program')['improvement'].agg(['mean', 'max', 'min'])
        
        x = range(len(program_performance))
        ax5.bar(x, program_performance['mean'], alpha=0.6, label='Average')
        ax5.scatter(x, program_performance['max'], color='green', s=50, label='Best', alpha=0.8)
        ax5.scatter(x, program_performance['min'], color='red', s=50, label='Worst', alpha=0.8)
        
        ax5.set_xlabel('Programs')
        ax5.set_ylabel('Efficiency Improvement (%)')
        ax5.set_title('Optimization Results by Program')
        ax5.set_xticks(x)
        ax5.set_xticklabels(program_performance.index, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        
        # 6. Strategy success rate
        ax6 = axes[2, 1]
        
        strategy_success = df.groupby('strategy').apply(
            lambda x: (x['improvement'] > 0).sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        bars = ax6.bar(range(len(strategy_success)), strategy_success.values, alpha=0.8)
        ax6.set_title('Strategy Success Rate (% of Programs Improved)')
        ax6.set_ylabel('Success Rate (%)')
        ax6.set_xticks(range(len(strategy_success)))
        ax6.set_xticklabels(strategy_success.index, rotation=45, ha='right')
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, value in zip(bars, strategy_success.values):
            ax6.text(bar.get_x() + bar.get_width()/2., value + 1,
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_optimization_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive optimization analysis report."""
        
        if df.empty:
            return "No optimization data available."
        
        # Calculate key statistics
        total_tests = len(df)
        successful_optimizations = len(df[df['improvement'] > 0])
        avg_improvement = df['improvement'].mean()
        best_improvement = df['improvement'].max()
        worst_degradation = df['improvement'].min()
        
        best_strategy = df.loc[df['improvement'].idxmax()]
        worst_strategy = df.loc[df['improvement'].idxmin()]
        
        report = f"""
ADVANCED OPTIMIZATION ANALYSIS REPORT
====================================

Analysis Overview:
- Total Strategy-Program Combinations: {total_tests}
- Successful Optimizations: {successful_optimizations} ({successful_optimizations/total_tests*100:.1f}%)
- Average Improvement: {avg_improvement:+.2f}%
- Best Result: {best_improvement:+.1f}% ({best_strategy['strategy']} on {best_strategy['program']})
- Worst Result: {worst_degradation:+.1f}% ({worst_strategy['strategy']} on {worst_strategy['program']})

STRATEGY CATEGORY PERFORMANCE:
-----------------------------
"""
        
        # Category analysis
        category_stats = df.groupby('category').agg({
            'improvement': ['mean', 'std', 'count'],
            'optimized_efficiency': 'mean'
        }).round(2)
        
        for category in category_stats.index:
            stats = category_stats.loc[category]
            success_rate = (df[df['category'] == category]['improvement'] > 0).mean() * 100
            
            report += f"\n{category.upper()}:\n"
            report += f"  Average Improvement: {stats[('improvement', 'mean')]:+.1f}% ± {stats[('improvement', 'std')]:.1f}%\n"
            report += f"  Success Rate: {success_rate:.1f}%\n"
            report += f"  Tests Conducted: {stats[('improvement', 'count')]}\n"
            report += f"  Average Final Efficiency: {stats[('optimized_efficiency', 'mean')]:.1f}%\n"
        
        # Top performing strategies
        top_strategies = df.groupby('strategy')['improvement'].mean().sort_values(ascending=False).head(5)
        
        report += f"\nTOP 5 OPTIMIZATION STRATEGIES:\n"
        report += f"------------------------------\n"
        for i, (strategy, improvement) in enumerate(top_strategies.items()):
            success_rate = (df[df['strategy'] == strategy]['improvement'] > 0).mean() * 100
            report += f"{i+1}. {strategy}: {improvement:+.1f}% avg improvement ({success_rate:.0f}% success rate)\n"
        
        # Bottom performing strategies
        bottom_strategies = df.groupby('strategy')['improvement'].mean().sort_values(ascending=True).head(3)
        
        report += f"\nLEAST EFFECTIVE STRATEGIES:\n"
        report += f"---------------------------\n"
        for i, (strategy, improvement) in enumerate(bottom_strategies.items()):
            report += f"{i+1}. {strategy}: {improvement:+.1f}% avg improvement\n"
        
        # Program-specific insights
        program_stats = df.groupby('program').agg({
            'improvement': ['mean', 'max', 'min'],
            'baseline_efficiency': 'first'
        }).round(1)
        
        report += f"\nPROGRAM-SPECIFIC ANALYSIS:\n"
        report += f"-------------------------\n"
        for program in program_stats.index:
            stats = program_stats.loc[program]
            baseline = stats[('baseline_efficiency', 'first')]
            avg_imp = stats[('improvement', 'mean')]
            best_imp = stats[('improvement', 'max')]
            
            report += f"{program}:\n"
            report += f"  Baseline Efficiency: {baseline:.1f}%\n"
            report += f"  Average Additional Improvement: {avg_imp:+.1f}%\n"
            report += f"  Best Additional Improvement: {best_imp:+.1f}%\n"
            report += f"  Potential Final Efficiency: {baseline + best_imp:.1f}%\n\n"
        
        # Key insights and recommendations
        compression_effective = df[df['category'] == 'compression']['improvement'].mean()
        hybrid_effective = df[df['category'] == 'hybrid']['improvement'].mean()
        encoding_effective = df[df['category'] == 'encoding']['improvement'].mean()
        semantic_effective = df[df['category'] == 'semantic']['improvement'].mean()
        
        report += f"KEY INSIGHTS:\n"
        report += f"============\n"
        report += f"1. Most Effective Category: {df.groupby('category')['improvement'].mean().idxmax()}\n"
        report += f"2. Compression Strategies: {compression_effective:+.1f}% average improvement\n"
        report += f"3. Hybrid Approaches: {hybrid_effective:+.1f}% average improvement\n"
        report += f"4. Encoding Optimizations: {encoding_effective:+.1f}% average improvement\n"
        report += f"5. Semantic Enhancements: {semantic_effective:+.1f}% average improvement\n"
        
        report += f"\nRECOMMENDATIONS:\n"
        report += f"===============\n"
        
        if successful_optimizations / total_tests > 0.5:
            report += f"✅ PROMISING: {successful_optimizations/total_tests*100:.0f}% of optimizations show improvement\n"
        else:
            report += f"⚠️  MIXED RESULTS: Only {successful_optimizations/total_tests*100:.0f}% of optimizations show improvement\n"
        
        if avg_improvement > 5:
            report += f"✅ HIGH IMPACT: Average {avg_improvement:.1f}% additional improvement beyond baseline\n"
        elif avg_improvement > 0:
            report += f"✅ MODERATE IMPACT: Average {avg_improvement:.1f}% additional improvement\n"
        else:
            report += f"❌ NEGATIVE IMPACT: Average {avg_improvement:.1f}% degradation - reconsider approaches\n"
        
        best_category = df.groupby('category')['improvement'].mean().idxmax()
        report += f"🎯 FOCUS AREA: Prioritize {best_category} strategies for best results\n"
        
        if best_improvement > 20:
            report += f"🚀 BREAKTHROUGH POTENTIAL: Best result of {best_improvement:.1f}% suggests major gains possible\n"
        
        return report
    
    def __del__(self):
        """Cleanup."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

def get_optimization_test_programs() -> List[Tuple[str, str, str, str]]:
    """Get test programs for optimization analysis."""
    
    return [
        ("simple_calc", 
         "#include <stdio.h>\nint main() { int a=5, b=3; printf(\"%d\\n\", a+b); return 0; }",
         "528000a0528000609900037f0b00002052800000d65f03c0",
         "mov w0, #0x5\nmov w1, #0x3\nstr wzr, [sp, #0xc]\nadd w0, w1, w0\nmov w0, #0x0\nret"),
        
        ("loop_sum",
         "#include <stdio.h>\nint main() { int sum=0; for(int i=1; i<=5; i++) sum+=i; printf(\"%d\\n\", sum); return 0; }",
         "52800020528000209900001f110004007100014c5400006c0b00002099000020531865fb7100014c54ffff8b52800000d65f03c0",
         "mov w0, #0x1\nmov w1, #0x1\nstr wzr, [sp]\nadd w0, w0, #0x1\ncmp w0, #0x5\nb.le loop\nadd w1, w1, w0\nstr w1, [sp]\ncmp w0, #0x5\nb.lt loop\nmov w0, #0x0\nret"),
        
        ("factorial",
         "#include <stdio.h>\nint fact(int n) { return n<=1 ? 1 : n*fact(n-1); }\nint main() { printf(\"%d\\n\", fact(5)); return 0; }",
         "7100043f5400004c52800020d10004401b007c20d65f03c052800140aa0003e0940000015280000052800000d65f03c0",
         "cmp w0, #0x1\nb.le base\nmov w0, #0x1\nsub w0, w0, #0x1\nmul w0, w1, w0\nret\nbase: mov w0, #0x1\nret\nmov w0, #0x5\nmov x0, x0\nbl fact\nmov w0, #0x0\nret")
    ]

def main():
    """Run advanced optimization research."""
    
    print("ADVANCED OPTIMIZATION RESEARCH")
    print("=" * 40)
    
    # Initialize optimizer
    optimizer = AdvancedOptimizer("gpt-4")
    
    print(f"Testing {len(optimizer.strategies)} optimization strategies:")
    for strategy in optimizer.strategies:
        print(f"  - {strategy.name} ({strategy.category}): {strategy.description}")
    
    # Get test programs
    test_programs = get_optimization_test_programs()
    
    print(f"\nTest programs: {len(test_programs)}")
    for name, _, _, _ in test_programs:
        print(f"  - {name}")
    
    # Run optimization analysis
    results_df = optimizer.run_optimization_analysis(test_programs)
    
    if not results_df.empty:
        # Generate report
        report = optimizer.generate_optimization_report(results_df)
        print(report)
        
        # Create visualizations
        optimizer.create_optimization_visualizations(results_df)
        
        # Save results
        results_df.to_csv('optimization_analysis_results.csv', index=False)
        
        with open('optimization_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✅ Advanced optimization analysis complete!")
        print("Files generated:")
        print("- optimization_analysis_results.csv")
        print("- optimization_analysis.png")
        print("- optimization_report.txt")
        
    else:
        print("❌ No optimization data collected.")

if __name__ == "__main__":
    main()