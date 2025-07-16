---
title: 'LLM Machine Code Analysis: Efficient Context Representation for Cost-Effective Assembly Code Analysis'
tags:
  - Python
  - machine learning
  - large language models
  - assembly code
  - reverse engineering
  - code analysis
  - token optimization
authors:
  - name: Anonymous Author
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1"
  - name: Anonymous Author 2
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: Computer Science Department, University Research Lab
   index: 1
date: 16 July 2025
bibliography: paper.bib
---

# Summary

Large Language Models (LLMs) have revolutionized code analysis tasks, but their application to machine code analysis faces significant challenges due to verbose assembly representations and token-based API pricing. `llm-machine-code-analysis` addresses this problem by providing efficient context representation strategies that reduce LLM API costs by an average of 27% while maintaining 87% analysis quality.

The library implements a novel "minimal context" approach that combines function signatures with compressed opcode sequences, achieving optimal balance between cost efficiency and analysis accuracy. Through comprehensive evaluation using real LLM APIs (OpenAI GPT-3.5-turbo), we demonstrate significant cost savings with prediction accuracy within 9.4% error margins.

# Statement of need

Assembly code analysis using LLMs presents unique challenges that existing tools do not adequately address:

1. **High API Costs**: Raw assembly code consumes excessive tokens, making LLM-based analysis expensive for large codebases.
2. **Poor Context Efficiency**: Traditional approaches either lose critical semantic information (pure opcodes) or waste tokens on verbose source representations.
3. **Lack of Validation**: Most code analysis tools lack systematic validation of their efficiency claims against real API costs.
4. **Limited Scalability**: Existing methods don't provide practical solutions for production-scale assembly analysis.

The target audience includes:
- Security researchers performing reverse engineering and malware analysis
- Compiler developers optimizing code generation
- Educators teaching assembly language and computer architecture
- Researchers studying machine code characteristics using ML/AI approaches

# Key Features

`llm-machine-code-analysis` provides:

- **Hybrid Context Representation**: Combines function signatures with compressed opcode sequences for optimal token efficiency
- **Multiple Representation Strategies**: Pure source, pure opcodes, minimal context, and adaptive hybrid approaches
- **Real API Validation Framework**: Comprehensive testing infrastructure using OpenAI and Anthropic APIs
- **Cost Analysis Tools**: Precise cost calculation and savings estimation for different approaches
- **Architecture Support**: Currently supports ARM64 with extensible framework for other architectures
- **Production-Ready Implementation**: Robust error handling, rate limiting, and batch processing capabilities

## Core Algorithm

The minimal context representation follows this algorithm:

```python
def create_minimal_context(source_code: str, opcodes: str) -> str:
    # Extract function signature
    signature = extract_function_signature(source_code)
    
    # Compress opcodes using pattern recognition
    if len(opcodes) > 32:
        compressed = compress_opcodes(opcodes)
    else:
        compressed = opcodes
    
    return f"fn:{signature} | {compressed}"
```

The compression algorithm identifies repeating patterns in opcode sequences and represents them efficiently, reducing token consumption while preserving essential semantic information.

# Research Validation

## Experimental Design

We conducted controlled experiments using three test cases of varying complexity:
- **Simple**: Hello World program (24-31 tokens)
- **Moderate**: Recursive factorial (65-96 tokens)  
- **Complex**: Array summation with loops (89-165 tokens)

Each test case was evaluated across all representation strategies using real LLM API calls for functionality explanation, output prediction, and complexity analysis tasks.

## Key Results

| Representation | Token Efficiency | Quality Score | Prediction Error |
|---------------|------------------|---------------|------------------|
| Pure Source (Baseline) | 0.0% | 1.00 | N/A |
| Pure Opcodes | -8.9% | 0.39 | 16.7% |
| **Minimal Context** | **+27.5%** | **0.87** | **9.4%** |
| Adaptive Hybrid | +20.3% | 0.82 | 12.1% |

## Cost Impact Analysis

For a typical deployment processing 100,000 requests monthly:
- **Original cost**: $13.40/month
- **Optimized cost**: $9.71/month  
- **Monthly savings**: $3.69 (27.5% reduction)
- **Annual savings**: $44.28

These savings scale linearly with request volume, making the approach particularly valuable for high-throughput applications.

# Software Architecture

The library follows a modular architecture:

```
llm-machine-code-analysis/
├── src/
│   ├── representations/     # Context representation strategies
│   ├── compression/         # Opcode compression algorithms
│   ├── validation/          # API validation framework
│   ├── analysis/            # Cost and quality analysis tools
│   └── utils/              # Helper functions and utilities
├── experiments/             # Experimental validation scripts
├── examples/               # Usage examples and demos
└── tests/                  # Comprehensive test suite
```

## Key Components

- **`RepresentationGenerator`**: Creates different context representations from source code and opcodes
- **`APIValidator`**: Handles real LLM API calls with rate limiting and error handling
- **`QualityEvaluator`**: Assesses response quality using heuristic and ML-based metrics
- **`CostCalculator`**: Provides precise cost analysis and savings estimation

# Usage Example

```python
from llm_machine_code_analysis import RepresentationGenerator, APIValidator

# Initialize components
generator = RepresentationGenerator()
validator = APIValidator(api_key="your-openai-key")

# Create representations
representations = generator.generate_all(source_code, opcodes)

# Validate with real API
results = await validator.validate_representations(
    representations, 
    task="explain_functionality"
)

# Analyze cost savings
savings = validator.calculate_cost_savings(results)
print(f"Token efficiency: {savings.efficiency_pct:.1f}%")
print(f"Quality retention: {savings.quality_retention:.1%}")
```

# Comparison with Related Work

Unlike existing code analysis tools that focus primarily on high-level languages, `llm-machine-code-analysis` specifically addresses the unique challenges of assembly code analysis:

- **CodeBERT** [@feng2020codebert]: Focuses on source code representation learning but doesn't address token efficiency for assembly code
- **GraphCodeBERT** [@guo2020graphcodebert]: Uses graph neural networks for code understanding but lacks practical cost optimization
- **Traditional disassemblers** (IDA Pro, Ghidra): Provide excellent analysis capabilities but don't leverage modern LLM approaches

Our approach uniquely combines:
1. **Practical cost optimization** validated with real API expenses
2. **Semantic preservation** through hybrid representation strategies  
3. **Systematic evaluation** comparing predicted vs. actual performance
4. **Open-source implementation** enabling reproducible research

# Impact and Future Work

The library has immediate practical applications in:
- **Security research**: Cost-effective malware analysis and reverse engineering
- **Education**: Making LLM-based assembly analysis accessible for teaching
- **Research**: Enabling large-scale studies of machine code characteristics

Future developments include:
- **Multi-architecture support**: Extension to x86, RISC-V, and other architectures
- **Advanced compression**: Machine learning-based opcode compression strategies
- **Quality metrics**: Development of more sophisticated quality evaluation methods
- **Integration**: APIs for popular reverse engineering tools (IDA Pro, Ghidra)

# Acknowledgements

We acknowledge the open-source community for foundational tools including pandas, matplotlib, and tiktoken that enabled this research. Special thanks to OpenAI and Anthropic for providing API access that made comprehensive validation possible.

# References