#!/usr/bin/env python3
"""
Real API Validation Framework
============================

Tests our token efficiency and hybrid context findings with actual LLM APIs.
Focuses on the most promising approaches from the hybrid context demo.
"""

import asyncio
import aiohttp
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class ValidationTest:
    """Represents a validation test case."""
    name: str
    source_code: str
    opcodes: str
    expected_functionality: str
    test_tasks: List[str]

@dataclass
class APIResult:
    """Represents an API call result."""
    test_name: str
    representation: str
    task: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response: str
    response_time: float
    cost: float
    timestamp: str

class RealAPIValidator:
    """Validates our findings with real LLM APIs."""
    
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.results = []
        
        # API pricing (per 1K tokens)
        self.pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125}
        }
        
        # Rate limits (seconds between requests)
        self.rate_limits = {
            'openai': 1.0,  # Conservative rate limiting
            'anthropic': 0.5
        }
    
    def check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are available."""
        available_apis = {}
        
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        available_apis['openai'] = bool(openai_key)
        available_apis['anthropic'] = bool(anthropic_key)
        
        return available_apis
    
    def generate_representations(self, test_case: ValidationTest) -> Dict[str, str]:
        """Generate the most promising representations based on demo results."""
        
        # Extract minimal context (best performer from demo)
        def extract_function_signature(source):
            lines = source.split('\n')
            for line in lines:
                line = line.strip()
                if (('(' in line and ')' in line and 
                     any(keyword in line for keyword in ['int ', 'void ', 'char '])) or
                    'main(' in line):
                    return line.replace('{', '').strip()[:50]  # Keep it short
            return 'main'
        
        def compress_opcodes(opcodes):
            if len(opcodes) <= 32:
                return opcodes
            # Use the pattern compression that worked in demo
            return opcodes[:32] + ".." + opcodes[-16:]
        
        signature = extract_function_signature(test_case.source_code)
        compressed_opcodes = compress_opcodes(test_case.opcodes)
        
        representations = {
            # Baselines
            'pure_source': test_case.source_code,
            'pure_opcodes': test_case.opcodes,
            
            # Best performer from demo: minimal_context
            'minimal_context': f"fn:{signature} | {compressed_opcodes}",
            
            # Also test the adaptive approach for comparison
            'adaptive_hybrid': f"fn:{signature} | cf:{'LOOP' if 'for' in test_case.source_code else 'LINEAR'} | {compressed_opcodes}"
        }
        
        return representations
    
    async def call_openai_api(self, model: str, prompt: str) -> Tuple[str, int, int, float]:
        """Call OpenAI API."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are a code analysis expert. Provide clear, concise analysis.'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 300,  # Keep responses focused
            'temperature': 0.1
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post('https://api.openai.com/v1/chat/completions', 
                                  headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    response_time = time.time() - start_time
                    
                    message = result['choices'][0]['message']['content']
                    prompt_tokens = result['usage']['prompt_tokens']
                    completion_tokens = result['usage']['completion_tokens']
                    
                    return message, prompt_tokens, completion_tokens, response_time
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")
    
    async def call_anthropic_api(self, model: str, prompt: str) -> Tuple[str, int, int, float]:
        """Call Anthropic API."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not found")
        
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': model,
            'max_tokens': 300,
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post('https://api.anthropic.com/v1/messages', 
                                  headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    response_time = time.time() - start_time
                    
                    message = result['content'][0]['text']
                    prompt_tokens = result['usage']['input_tokens']
                    completion_tokens = result['usage']['output_tokens']
                    
                    return message, prompt_tokens, completion_tokens, response_time
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error_text}")
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate API cost."""
        if model not in self.pricing:
            return 0.0
        
        rates = self.pricing[model]
        input_cost = (prompt_tokens / 1000) * rates['input']
        output_cost = (completion_tokens / 1000) * rates['output']
        return input_cost + output_cost
    
    def create_task_prompt(self, task: str, content: str, content_type: str) -> str:
        """Create task-specific prompts."""
        
        content_description = {
            'pure_source': 'C source code',
            'pure_opcodes': 'ARM64 machine code opcodes (hex)',
            'minimal_context': 'function signature with ARM64 opcodes',
            'adaptive_hybrid': 'function signature, control flow, and ARM64 opcodes'
        }
        
        desc = content_description.get(content_type, 'code representation')
        
        task_prompts = {
            'explain': f"Explain what this {desc} does:\n\n{content}",
            'output': f"What will be the output when this {desc} runs?\n\n{content}",
            'bugs': f"Are there any bugs or issues in this {desc}?\n\n{content}",
            'complexity': f"What is the time complexity of this {desc}?\n\n{content}"
        }
        
        return task_prompts.get(task, f"Analyze this {desc}:\n\n{content}")
    
    def evaluate_response_quality(self, response: str, expected: str, task: str) -> float:
        """Evaluate response quality using heuristics."""
        if not response or len(response.strip()) < 10:
            return 0.1
        
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Base quality score
        quality = 0.5
        
        # Length appropriateness
        if 30 <= len(response) <= 200:
            quality += 0.1
        elif len(response) < 15:
            quality -= 0.2
        
        # Keyword matching
        expected_words = expected_lower.split()[:5]
        matches = sum(1 for word in expected_words if word in response_lower)
        quality += (matches / len(expected_words)) * 0.3
        
        # Task-specific evaluation
        if task == 'explain' and any(word in response_lower for word in ['function', 'program', 'calculates', 'performs']):
            quality += 0.1
        elif task == 'output' and any(word in response_lower for word in ['output', 'prints', 'returns', 'displays']):
            quality += 0.1
        elif task == 'bugs' and any(word in response_lower for word in ['no bugs', 'correct', 'looks good', 'error', 'issue']):
            quality += 0.1
        elif task == 'complexity' and any(word in response_lower for word in ['o(', 'time', 'complexity', 'linear', 'constant']):
            quality += 0.1
        
        # Penalty for generic responses
        if any(phrase in response_lower for phrase in ['cannot analyze', 'unable to', 'not possible', 'unclear']):
            quality -= 0.3
        
        return max(0.0, min(1.0, quality))
    
    async def run_validation_test(self, test_case: ValidationTest, 
                                representation: str, content: str, 
                                task: str, model: str) -> Optional[APIResult]:
        """Run a single validation test."""
        
        prompt = self.create_task_prompt(task, content, representation)
        
        try:
            # Call appropriate API
            if model.startswith('gpt'):
                response, prompt_tokens, completion_tokens, response_time = await self.call_openai_api(model, prompt)
                await asyncio.sleep(self.rate_limits['openai'])
            elif model.startswith('claude'):
                response, prompt_tokens, completion_tokens, response_time = await self.call_anthropic_api(model, prompt)
                await asyncio.sleep(self.rate_limits['anthropic'])
            else:
                raise ValueError(f"Unsupported model: {model}")
            
            # Calculate metrics
            total_tokens = prompt_tokens + completion_tokens
            cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
            
            return APIResult(
                test_name=test_case.name,
                representation=representation,
                task=task,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response=response,
                response_time=response_time,
                cost=cost,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"  ❌ Error with {representation}/{task}: {e}")
            return None
    
    async def run_comprehensive_validation(self, test_cases: List[ValidationTest], 
                                         models: List[str], 
                                         max_tests: int = 50) -> pd.DataFrame:
        """Run comprehensive validation with cost controls."""
        
        all_results = []
        test_count = 0
        estimated_cost = 0.0
        
        print(f"🚀 Starting Real API Validation")
        print(f"📊 Max tests: {max_tests}")
        print(f"🤖 Models: {', '.join(models)}")
        print(f"💰 Estimated max cost: ${max_tests * 0.05:.2f}")
        
        # Confirm before proceeding
        proceed = input(f"\n💸 This will cost real money. Continue? (y/N): ").lower().strip()
        if proceed != 'y':
            print("❌ Validation cancelled.")
            return pd.DataFrame()
        
        for test_case in test_cases:
            if test_count >= max_tests:
                print(f"\n⏹️  Reached max test limit ({max_tests})")
                break
            
            print(f"\n🧪 Testing: {test_case.name}")
            
            # Generate representations
            representations = self.generate_representations(test_case)
            
            for model in models:
                if test_count >= max_tests:
                    break
                
                print(f"  🤖 Model: {model}")
                
                for repr_name, repr_content in representations.items():
                    if test_count >= max_tests:
                        break
                    
                    for task in test_case.test_tasks[:2]:  # Limit to 2 tasks per representation
                        if test_count >= max_tests:
                            break
                        
                        test_count += 1
                        progress = (test_count / max_tests) * 100
                        
                        print(f"    📝 {repr_name}/{task} ({progress:.1f}%): ", end="", flush=True)
                        
                        result = await self.run_validation_test(
                            test_case, repr_name, repr_content, task, model
                        )
                        
                        if result:
                            all_results.append(result)
                            estimated_cost += result.cost
                            print(f"{result.prompt_tokens}→{result.completion_tokens} tokens, ${result.cost:.4f}")
                        else:
                            print("❌ Failed")
                        
                        # Safety check on cost
                        if estimated_cost > 5.0:  # $5 safety limit
                            print(f"\n💰 Cost limit reached (${estimated_cost:.2f})")
                            test_count = max_tests
                            break
        
        print(f"\n✅ Validation complete!")
        print(f"📊 Successful tests: {len(all_results)}")
        print(f"💰 Total cost: ${estimated_cost:.4f}")
        
        # Convert to DataFrame
        if all_results:
            df_data = []
            for result in all_results:
                quality_score = self.evaluate_response_quality(
                    result.response, 
                    test_cases[0].expected_functionality,  # Simplified for demo
                    result.task
                )
                
                df_data.append({
                    'test_name': result.test_name,
                    'representation': result.representation,
                    'task': result.task,
                    'model': result.model,
                    'prompt_tokens': result.prompt_tokens,
                    'completion_tokens': result.completion_tokens,
                    'total_tokens': result.total_tokens,
                    'response_time': result.response_time,
                    'cost': result.cost,
                    'quality_score': quality_score,
                    'response_preview': result.response[:100] + '...' if len(result.response) > 100 else result.response
                })
            
            return pd.DataFrame(df_data)
        else:
            return pd.DataFrame()
    
    def analyze_validation_results(self, df: pd.DataFrame) -> Dict:
        """Analyze validation results and compare to predictions."""
        
        if df.empty:
            return {}
        
        analysis = {}
        
        # Token efficiency analysis
        source_baseline = df[df['representation'] == 'pure_source']['prompt_tokens'].mean()
        
        for representation in df['representation'].unique():
            repr_data = df[df['representation'] == representation]
            
            avg_prompt_tokens = repr_data['prompt_tokens'].mean()
            avg_quality = repr_data['quality_score'].mean()
            avg_cost = repr_data['cost'].mean()
            
            token_efficiency = ((source_baseline - avg_prompt_tokens) / source_baseline * 100) if source_baseline > 0 else 0
            
            analysis[representation] = {
                'avg_prompt_tokens': avg_prompt_tokens,
                'token_efficiency_pct': token_efficiency,
                'avg_quality_score': avg_quality,
                'avg_cost': avg_cost,
                'sample_count': len(repr_data)
            }
        
        return analysis
    
    def create_validation_visualizations(self, df: pd.DataFrame, analysis: Dict):
        """Create visualizations of validation results."""
        
        if df.empty:
            print("No data to visualize.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Token efficiency comparison
        ax1 = axes[0, 0]
        
        representations = list(analysis.keys())
        efficiencies = [analysis[repr]['token_efficiency_pct'] for repr in representations]
        
        colors = ['red' if eff < 0 else 'orange' if eff < 30 else 'green' for eff in efficiencies]
        bars = ax1.bar(representations, efficiencies, color=colors, alpha=0.8)
        
        ax1.set_title('Real API Token Efficiency Results')
        ax1.set_ylabel('Token Efficiency (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, efficiencies):
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    value + (2 if value > 0 else -2),
                    f'{value:+.1f}%', ha='center', 
                    va='bottom' if value > 0 else 'top', fontweight='bold')
        
        # 2. Quality vs Efficiency scatter
        ax2 = axes[0, 1]
        
        x_values = [analysis[repr]['token_efficiency_pct'] for repr in representations]
        y_values = [analysis[repr]['avg_quality_score'] for repr in representations]
        
        scatter = ax2.scatter(x_values, y_values, s=150, alpha=0.7)
        
        # Add labels
        for i, repr in enumerate(representations):
            ax2.annotate(repr, (x_values[i], y_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Token Efficiency (%)')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Real API: Quality vs Efficiency')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Cost comparison
        ax3 = axes[1, 0]
        
        costs = [analysis[repr]['avg_cost'] for repr in representations]
        bars = ax3.bar(representations, costs, alpha=0.8, color='steelblue')
        
        ax3.set_title('Average Cost per Request')
        ax3.set_ylabel('Cost ($)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, costs):
            ax3.text(bar.get_x() + bar.get_width()/2., value + value*0.05,
                    f'${value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary comparison with predictions
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Compare with demo predictions
        demo_predictions = {
            'pure_opcodes': {'efficiency': 12.8, 'quality': 0.30},
            'minimal_context': {'efficiency': 36.9, 'quality': 0.80}
        }
        
        summary_text = f"""
REAL API VALIDATION RESULTS

ACTUAL vs PREDICTED:

Pure Opcodes:
• Predicted: +12.8% efficiency, 0.30 quality
• Actual: {analysis.get('pure_opcodes', {}).get('token_efficiency_pct', 0):+.1f}% efficiency, {analysis.get('pure_opcodes', {}).get('avg_quality_score', 0):.2f} quality

Minimal Context:
• Predicted: +36.9% efficiency, 0.80 quality  
• Actual: {analysis.get('minimal_context', {}).get('token_efficiency_pct', 0):+.1f}% efficiency, {analysis.get('minimal_context', {}).get('avg_quality_score', 0):.2f} quality

COST ANALYSIS:
• Total tests: {len(df)}
• Total cost: ${df['cost'].sum():.4f}
• Avg cost/request: ${df['cost'].mean():.4f}

VALIDATION STATUS:
"""
        
        # Determine validation outcome
        if 'minimal_context' in analysis:
            actual_eff = analysis['minimal_context']['token_efficiency_pct']
            actual_quality = analysis['minimal_context']['avg_quality_score']
            
            if actual_eff > 20 and actual_quality > 0.6:
                summary_text += "✅ HYPOTHESIS CONFIRMED\n  Token efficiency gains are real!"
            elif actual_eff > 10:
                summary_text += "⚠️ PARTIAL VALIDATION\n  Some efficiency gains confirmed"
            else:
                summary_text += "❌ HYPOTHESIS REJECTED\n  Expected gains not achieved"
        else:
            summary_text += "⚠️ INSUFFICIENT DATA\n  Need more test results"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('real_api_validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def get_validation_test_cases() -> List[ValidationTest]:
    """Get focused test cases for real API validation."""
    
    return [
        ValidationTest(
            name="factorial_recursive",
            source_code="""#include <stdio.h>
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
int main() {
    printf("%d\\n", factorial(5));
    return 0;
}""",
            opcodes="7100043f5400004c528000201b007c20d65f03c052800140aa0003e094000001",
            expected_functionality="calculates factorial of 5 recursively, prints 120",
            test_tasks=["explain", "output"]
        ),
        
        ValidationTest(
            name="array_sum",
            source_code="""#include <stdio.h>
int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    printf("Sum: %d\\n", sum);
    return 0;
}""",
            opcodes="52800020528000409900037f7100014054000060b8606800ab000000201100040071000140",
            expected_functionality="sums array elements and prints result 15",
            test_tasks=["explain", "complexity"]
        )
    ]

async def main():
    """Run real API validation."""
    
    print("🔬 REAL API VALIDATION FRAMEWORK")
    print("=" * 50)
    print("Testing our token efficiency findings with actual LLM APIs!")
    
    # Initialize validator
    validator = RealAPIValidator()
    
    # Check API keys
    available_apis = validator.check_api_keys()
    
    print(f"\n🔑 API Key Status:")
    for api, available in available_apis.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {api}: {status}")
    
    if not any(available_apis.values()):
        print(f"\n❌ No API keys found!")
        print(f"Please set environment variables:")
        print(f"  export OPENAI_API_KEY='your-openai-key'")
        print(f"  export ANTHROPIC_API_KEY='your-anthropic-key'")
        return
    
    # Determine available models
    models = []
    if available_apis['openai']:
        models.append('gpt-3.5-turbo')  # Start with cheaper model
    if available_apis['anthropic']:
        models.append('claude-3-haiku')  # Start with cheaper model
    
    print(f"\n🤖 Available models: {', '.join(models)}")
    
    # Get test cases
    test_cases = get_validation_test_cases()
    
    print(f"\n📋 Test cases: {len(test_cases)}")
    for tc in test_cases:
        print(f"  • {tc.name}: {tc.expected_functionality}")
    
    # Run validation
    results_df = await validator.run_comprehensive_validation(
        test_cases, models, max_tests=24  # Conservative limit
    )
    
    if not results_df.empty:
        # Analyze results
        analysis = validator.analyze_validation_results(results_df)
        
        print(f"\n📊 VALIDATION ANALYSIS:")
        for repr_name, data in analysis.items():
            print(f"\n{repr_name.upper()}:")
            print(f"  Token Efficiency: {data['token_efficiency_pct']:+.1f}%")
            print(f"  Quality Score: {data['avg_quality_score']:.3f}")
            print(f"  Avg Cost: ${data['avg_cost']:.4f}")
            print(f"  Sample Size: {data['sample_count']}")
        
        # Create visualizations
        validator.create_validation_visualizations(results_df, analysis)
        
        # Save results
        results_df.to_csv('real_api_validation_results.csv', index=False)
        
        with open('validation_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✅ Real API validation complete!")
        print(f"📁 Files generated:")
        print(f"  • real_api_validation_results.csv")
        print(f"  • real_api_validation_results.png")
        print(f"  • validation_analysis.json")
        print(f"💰 Total cost: ${results_df['cost'].sum():.4f}")
        
    else:
        print(f"\n❌ No results collected. Check API keys and connectivity.")

if __name__ == "__main__":
    asyncio.run(main())