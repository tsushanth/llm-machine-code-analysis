#!/usr/bin/env python3
"""
LLM Performance Testing Framework
=================================

Tests whether LLMs can perform code-related tasks as well with opcodes
as they do with source code. This is critical for validating the practical
utility of our token efficiency gains.
"""

import os
import asyncio
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import tiktoken

@dataclass
class TestCase:
    """Represents a single test case for LLM evaluation."""
    name: str
    source_code: str
    opcodes: str
    task_type: str
    expected_output: str
    difficulty: str
    
@dataclass 
class LLMResponse:
    """Represents an LLM response to a test case."""
    test_case_name: str
    representation: str  # 'source' or 'opcode'
    task_type: str
    response: str
    tokens_used: int
    response_time: float
    cost_estimate: float

class LLMPerformanceTester:
    """Framework for testing LLM performance on code tasks."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.results = []
        
        # Token costs (per 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }
    
    def create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases for evaluation."""
        
        test_cases = []
        
        # Simple arithmetic program
        simple_source = '''#include <stdio.h>
int main() {
    int a = 10, b = 5;
    printf("Sum: %d\\n", a + b);
    printf("Product: %d\\n", a * b);
    return 0;
}'''
        simple_opcodes = "52800140528000a0b900037f0b000020b90003bf52800000d65f03c0"
        
        test_cases.extend([
            TestCase("simple_arithmetic", simple_source, simple_opcodes, 
                    "explanation", "This program adds and multiplies two integers", "easy"),
            TestCase("simple_arithmetic", simple_source, simple_opcodes,
                    "output_prediction", "Sum: 15\\nProduct: 50", "easy"),
            TestCase("simple_arithmetic", simple_source, simple_opcodes,
                    "bug_detection", "No bugs found", "easy"),
        ])
        
        # Loop program
        loop_source = '''#include <stdio.h>
int main() {
    int sum = 0;
    for (int i = 1; i <= 5; i++) {
        sum += i;
    }
    printf("Sum: %d\\n", sum);
    return 0;
}'''
        loop_opcodes = "52800020528000209900001f110004007100014c5400006c0b00002099000020531865fb7100014c54ffff8b"
        
        test_cases.extend([
            TestCase("loop_sum", loop_source, loop_opcodes,
                    "explanation", "This program calculates the sum of numbers 1 to 5 using a loop", "medium"),
            TestCase("loop_sum", loop_source, loop_opcodes,
                    "output_prediction", "Sum: 15", "medium"),
            TestCase("loop_sum", loop_source, loop_opcodes,
                    "optimization", "The loop could be replaced with formula: n*(n+1)/2", "medium"),
        ])
        
        # Function with bug
        buggy_source = '''#include <stdio.h>
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);  // Missing base case for n=0
}
int main() {
    printf("5! = %d\\n", factorial(5));
    return 0;
}'''
        buggy_opcodes = "7100043f5400004c528000201b007c0052800020d10004401b007c20d65f03c0"
        
        test_cases.extend([
            TestCase("factorial_bug", buggy_source, buggy_opcodes,
                    "bug_detection", "The base case handles n<=1, but factorial(0) should be 1, which it does", "hard"),
            TestCase("factorial_bug", buggy_source, buggy_opcodes,
                    "explanation", "This is a recursive factorial function", "medium"),
        ])
        
        return test_cases
    
    def create_prompts(self, test_case: TestCase, representation: str) -> str:
        """Create task-specific prompts for source code or opcodes."""
        
        if representation == "source":
            code_block = f"```c\n{test_case.source_code}\n```"
            code_type = "C source code"
        else:  # opcodes
            code_block = f"```hex\n{test_case.opcodes}\n```"
            code_type = "ARM64 machine code opcodes"
        
        prompts = {
            "explanation": f"Explain what this {code_type} does:\n\n{code_block}",
            
            "output_prediction": f"What will be the output when this {code_type} runs?\n\n{code_block}",
            
            "bug_detection": f"Are there any bugs or issues in this {code_type}?\n\n{code_block}",
            
            "optimization": f"How could this {code_type} be optimized for better performance?\n\n{code_block}",
            
            "complexity": f"What is the time complexity of this {code_type}?\n\n{code_block}",
        }
        
        return prompts.get(test_case.task_type, f"Analyze this {code_type}:\n\n{code_block}")
    
    async def query_llm_mock(self, prompt: str) -> Tuple[str, float]:
        """Mock LLM query for testing (replace with actual API calls)."""
        
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        # Count tokens
        input_tokens = len(self.tokenizer.encode(prompt))
        
        # Mock responses based on content
        if "opcodes" in prompt.lower() or "machine code" in prompt.lower():
            if "explain" in prompt.lower():
                response = "This appears to be ARM64 machine code. The opcodes represent a sequence of instructions that perform arithmetic operations and system calls."
            elif "output" in prompt.lower():
                response = "Based on the instruction patterns, this code likely produces numerical output, but precise prediction requires execution context."
            elif "bug" in prompt.lower():
                response = "Without source context, it's difficult to identify logical bugs in machine code, but the instruction sequence appears syntactically valid."
            else:
                response = "This is ARM64 machine code consisting of arithmetic and control flow instructions."
        else:  # source code
            if "explain" in prompt.lower():
                response = "This C program performs arithmetic calculations, uses loops or conditionals, and prints results to stdout."
            elif "output" in prompt.lower():
                response = "The program will output numerical results based on the calculations performed."
            elif "bug" in prompt.lower():
                response = "The code appears to be syntactically correct, though there may be logical issues depending on requirements."
            else:
                response = "This is a C program that demonstrates basic programming concepts."
        
        output_tokens = len(self.tokenizer.encode(response))
        response_time = 0.5 + (input_tokens + output_tokens) * 0.001  # Simulate processing time
        
        return response, response_time
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated API cost."""
        if self.model_name not in self.pricing:
            return 0.0
        
        rates = self.pricing[self.model_name]
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        return input_cost + output_cost
    
    async def run_single_test(self, test_case: TestCase, representation: str) -> LLMResponse:
        """Run a single test case."""
        
        prompt = self.create_prompts(test_case, representation)
        
        # Count input tokens
        input_tokens = len(self.tokenizer.encode(prompt))
        
        # Query LLM (mock for now - replace with actual API)
        start_time = time.time()
        response, response_time = await self.query_llm_mock(prompt)
        
        # Count output tokens
        output_tokens = len(self.tokenizer.encode(response))
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        return LLMResponse(
            test_case_name=test_case.name,
            representation=representation,
            task_type=test_case.task_type,
            response=response,
            tokens_used=total_tokens,
            response_time=response_time,
            cost_estimate=cost
        )
    
    async def run_comprehensive_evaluation(self) -> pd.DataFrame:
        """Run comprehensive evaluation across all test cases."""
        
        test_cases = self.create_test_cases()
        all_results = []
        
        print(f"Running {len(test_cases)} test cases x 2 representations = {len(test_cases) * 2} total tests")
        
        for i, test_case in enumerate(test_cases):
            print(f"Testing {i+1}/{len(test_cases)}: {test_case.name} ({test_case.task_type})")
            
            # Test both representations
            for representation in ["source", "opcode"]:
                try:
                    result = await self.run_single_test(test_case, representation)
                    all_results.append(result)
                    print(f"  {representation}: {result.tokens_used} tokens, ${result.cost_estimate:.4f}")
                except Exception as e:
                    print(f"  Error testing {representation}: {e}")
        
        # Convert to DataFrame
        results_data = []
        for result in all_results:
            results_data.append({
                'test_case': result.test_case_name,
                'representation': result.representation,
                'task_type': result.task_type,
                'tokens_used': result.tokens_used,
                'response_time': result.response_time,
                'cost_estimate': result.cost_estimate,
                'response_length': len(result.response)
            })
        
        return pd.DataFrame(results_data)
    
    def evaluate_response_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate response quality metrics."""
        
        # Add quality metrics
        quality_scores = []
        
        for _, row in df.iterrows():
            # Mock quality scoring (in real implementation, use human evaluation or automated metrics)
            base_score = 0.8  # Assume good baseline
            
            # Adjust based on representation
            if row['representation'] == 'opcode':
                # Opcodes might be slightly less accurate
                quality_adjustment = -0.1
            else:
                quality_adjustment = 0.0
            
            # Adjust based on task complexity
            task_difficulty = {
                'explanation': 0.0,
                'output_prediction': -0.05,
                'bug_detection': -0.15,
                'optimization': -0.1,
                'complexity': -0.08
            }
            
            difficulty_adjustment = task_difficulty.get(row['task_type'], 0.0)
            
            final_score = base_score + quality_adjustment + difficulty_adjustment
            final_score = max(0.0, min(1.0, final_score))  # Clamp to [0,1]
            
            quality_scores.append(final_score)
        
        df['quality_score'] = quality_scores
        return df
    
    def create_performance_visualizations(self, df: pd.DataFrame):
        """Create comprehensive performance comparison visualizations."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Token usage comparison
        ax1 = axes[0, 0]
        token_comparison = df.groupby(['representation', 'task_type'])['tokens_used'].mean().unstack()
        token_comparison.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Token Usage by Task Type')
        ax1.set_ylabel('Tokens Used')
        ax1.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Cost comparison
        ax2 = axes[0, 1]
        cost_comparison = df.groupby(['representation', 'task_type'])['cost_estimate'].mean().unstack()
        cost_comparison.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Cost by Task Type')
        ax2.set_ylabel('Cost ($)')
        ax2.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Quality scores
        ax3 = axes[0, 2]
        quality_comparison = df.groupby(['representation', 'task_type'])['quality_score'].mean().unstack()
        quality_comparison.plot(kind='bar', ax=ax3)
        ax3.set_title('Average Quality Score by Task Type')
        ax3.set_ylabel('Quality Score (0-1)')
        ax3.set_ylim(0, 1)
        ax3.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Token efficiency vs quality tradeoff
        ax4 = axes[1, 0]
        source_data = df[df['representation'] == 'source']
        opcode_data = df[df['representation'] == 'opcode']
        
        ax4.scatter(source_data['tokens_used'], source_data['quality_score'], 
                   label='Source Code', alpha=0.7, s=100)
        ax4.scatter(opcode_data['tokens_used'], opcode_data['quality_score'], 
                   label='Opcodes', alpha=0.7, s=100)
        ax4.set_xlabel('Tokens Used')
        ax4.set_ylabel('Quality Score')
        ax4.set_title('Token Efficiency vs Quality Tradeoff')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Response time comparison
        ax5 = axes[1, 1]
        time_comparison = df.groupby('representation')['response_time'].mean()
        bars = ax5.bar(time_comparison.index, time_comparison.values, 
                      color=['steelblue', 'orange'], alpha=0.8)
        ax5.set_title('Average Response Time')
        ax5.set_ylabel('Response Time (seconds)')
        
        # Add value labels
        for bar, value in zip(bars, time_comparison.values):
            ax5.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                    f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 6. Overall efficiency metrics
        ax6 = axes[1, 2]
        
        # Calculate efficiency metrics
        source_avg = df[df['representation'] == 'source'].groupby('task_type').agg({
            'tokens_used': 'mean',
            'cost_estimate': 'mean',
            'quality_score': 'mean'
        })
        
        opcode_avg = df[df['representation'] == 'opcode'].groupby('task_type').agg({
            'tokens_used': 'mean',
            'cost_estimate': 'mean',
            'quality_score': 'mean'
        })
        
        # Calculate percentage improvements
        token_improvement = ((source_avg['tokens_used'] - opcode_avg['tokens_used']) / source_avg['tokens_used'] * 100).mean()
        cost_improvement = ((source_avg['cost_estimate'] - opcode_avg['cost_estimate']) / source_avg['cost_estimate'] * 100).mean()
        quality_change = ((opcode_avg['quality_score'] - source_avg['quality_score']) / source_avg['quality_score'] * 100).mean()
        
        metrics = ['Token\nReduction', 'Cost\nReduction', 'Quality\nChange']
        values = [token_improvement, cost_improvement, quality_change]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax6.bar(metrics, values, color=colors, alpha=0.8)
        ax6.set_title('Overall Performance Changes\n(Opcodes vs Source)')
        ax6.set_ylabel('Percentage Change (%)')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2., 
                    value + (2 if value > 0 else -2),
                    f'{value:+.1f}%', ha='center', 
                    va='bottom' if value > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('llm_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive performance analysis report."""
        
        source_data = df[df['representation'] == 'source']
        opcode_data = df[df['representation'] == 'opcode']
        
        report = f"""
LLM PERFORMANCE ANALYSIS REPORT
==============================

Model: {self.model_name}
Test Cases: {len(df) // 2}
Total Evaluations: {len(df)}

SUMMARY METRICS:
---------------
                        Source Code    Opcodes      Change
Average Tokens:         {source_data['tokens_used'].mean():.1f}        {opcode_data['tokens_used'].mean():.1f}       {((opcode_data['tokens_used'].mean() - source_data['tokens_used'].mean()) / source_data['tokens_used'].mean() * 100):+.1f}%
Average Cost:           ${source_data['cost_estimate'].mean():.4f}      ${opcode_data['cost_estimate'].mean():.4f}     {((opcode_data['cost_estimate'].mean() - source_data['cost_estimate'].mean()) / source_data['cost_estimate'].mean() * 100):+.1f}%
Average Quality:        {source_data['quality_score'].mean():.3f}        {opcode_data['quality_score'].mean():.3f}       {((opcode_data['quality_score'].mean() - source_data['quality_score'].mean()) / source_data['quality_score'].mean() * 100):+.1f}%
Average Response Time:  {source_data['response_time'].mean():.2f}s        {opcode_data['response_time'].mean():.2f}s      {((opcode_data['response_time'].mean() - source_data['response_time'].mean()) / source_data['response_time'].mean() * 100):+.1f}%

TASK-SPECIFIC ANALYSIS:
----------------------
"""
        
        task_types = df['task_type'].unique()
        for task in task_types:
            task_source = source_data[source_data['task_type'] == task]
            task_opcode = opcode_data[opcode_data['task_type'] == task]
            
            if not task_source.empty and not task_opcode.empty:
                token_change = ((task_opcode['tokens_used'].mean() - task_source['tokens_used'].mean()) / task_source['tokens_used'].mean() * 100)
                quality_change = ((task_opcode['quality_score'].mean() - task_source['quality_score'].mean()) / task_source['quality_score'].mean() * 100)
                
                report += f"\n{task.upper()}:\n"
                report += f"  Token Change: {token_change:+.1f}%\n"
                report += f"  Quality Change: {quality_change:+.1f}%\n"
                report += f"  Source Quality: {task_source['quality_score'].mean():.3f}\n"
                report += f"  Opcode Quality: {task_opcode['quality_score'].mean():.3f}\n"
        
        # Key insights
        avg_token_savings = ((source_data['tokens_used'].mean() - opcode_data['tokens_used'].mean()) / source_data['tokens_used'].mean() * 100)
        avg_quality_impact = ((opcode_data['quality_score'].mean() - source_data['quality_score'].mean()) / source_data['quality_score'].mean() * 100)
        
        report += f"""
KEY INSIGHTS:
------------
1. TOKEN EFFICIENCY: {abs(avg_token_savings):.1f}% {'savings' if avg_token_savings > 0 else 'increase'} with opcodes
2. QUALITY IMPACT: {abs(avg_quality_impact):.1f}% {'improvement' if avg_quality_impact > 0 else 'degradation'} in response quality
3. COST IMPLICATIONS: {'Significant cost reduction' if avg_token_savings > 0 else 'Increased costs'} for high-volume applications

RECOMMENDATIONS:
---------------
"""
        
        if avg_token_savings > 0 and abs(avg_quality_impact) < 10:
            report += "✅ RECOMMENDED: Opcodes provide good token efficiency with acceptable quality impact\n"
        elif avg_token_savings > 0 and abs(avg_quality_impact) >= 10:
            report += "⚠️  CONDITIONAL: Opcodes save tokens but impact quality - use for cost-sensitive applications\n"
        else:
            report += "❌ NOT RECOMMENDED: Opcodes increase tokens without quality benefits\n"
        
        report += f"""
4. Best task types for opcodes: {', '.join([task for task in task_types if opcode_data[opcode_data['task_type'] == task]['quality_score'].mean() > 0.7])}
5. Challenging task types: {', '.join([task for task in task_types if opcode_data[opcode_data['task_type'] == task]['quality_score'].mean() < 0.6])}

FUTURE WORK:
-----------
- Test with larger, more complex programs
- Evaluate specialized models trained on assembly/machine code
- Develop hybrid approaches combining source + opcodes
- Conduct human expert evaluation of response quality
"""
        
        return report

def create_real_llm_integration():
    """Example of how to integrate with real LLM APIs."""
    
    integration_code = '''
# Real LLM Integration Example (OpenAI API)
import openai

async def query_real_llm(self, prompt: str) -> Tuple[str, float]:
    """Query actual LLM API (replace mock function)."""
    
    start_time = time.time()
    
    try:
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a code analysis expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        response_time = time.time() - start_time
        response_text = response.choices[0].message.content
        
        return response_text, response_time
        
    except Exception as e:
        print(f"API Error: {e}")
        return f"Error: {e}", 0.0

# Usage:
# Replace query_llm_mock with query_real_llm in the tester
# Add your OpenAI API key: openai.api_key = "your-key-here"
'''
    
    return integration_code

async def main():
    """Run LLM performance testing."""
    
    print("LLM PERFORMANCE TESTING FRAMEWORK")
    print("="*40)
    
    # Initialize tester
    tester = LLMPerformanceTester("gpt-4")
    
    print("🔬 Running comprehensive evaluation...")
    print("   (Using mock responses - replace with real API for production)")
    
    # Run evaluation
    results_df = await tester.run_comprehensive_evaluation()
    
    # Add quality evaluation
    results_df = tester.evaluate_response_quality(results_df)
    
    # Generate report
    report = tester.generate_performance_report(results_df)
    print(report)
    
    # Create visualizations
    tester.create_performance_visualizations(results_df)
    
    # Save results
    results_df.to_csv('llm_performance_results.csv', index=False)
    
    with open('llm_performance_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✅ Performance testing complete!")
    print("Files generated:")
    print("- llm_performance_results.csv")
    print("- llm_performance_analysis.png")
    print("- llm_performance_report.txt")
    
    # Show integration example
    print("\n📝 For real API integration:")
    print(create_real_llm_integration())

if __name__ == "__main__":
    asyncio.run(main())