---
title: 'Why Naive Binary Representation Fails for LLM Efficiency'
tags:
  - language models
  - GPT-4
  - binary encoding
  - prompt engineering
  - assembly
  - token efficiency
authors:
  - name: Sushanth Tiruvaipati
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2025-07-01
bibliography: paper.bib
---

# Summary

We investigate the impact of representing code in binary and assembly form when prompting large language models (LLMs). Contrary to the intuitive belief that machine code may be more efficient for models to process, our analysis across 60 code tasks and four prompt types reveals a staggering token inflation of over 14,000% for binary representations. We show that naive binary encoding leads to massive cost overheads, making it a poor choice for efficient LLM inference.

# Statement of Need

Token efficiency is a critical concern in LLM-based applications, especially in production workflows where cost and latency matter. Despite recent interest in representing code using low-level formats such as hexadecimal or assembly (often with the assumption that models might interpret them more compactly or precisely), no prior study has quantified their actual token-level and financial implications. This work fills that gap with rigorous empirical analysis and provides practical guidance for developers and researchers considering low-level representations in prompt design.

# Methodology

We built a pipeline that compiles C source code into binary and assembly formats. Each representation was then used to prompt GPT-4 across four tasks:

- Explanation
- Optimization
- Debugging
- Conversion

For each prompt type and representation, we measured:

- Token counts of input prompts.
- Token inflation or savings vs. the original source.
- Estimated API cost assuming 500 output tokens.

# Results

## Token Inflation

- **Binary representation**: +14,543% tokens on average.
- **Assembly representation**: +903% tokens on average.
- **Worst-case binary inflation**: +48,045%.

## Cost Impact

- **Source code prompts**: \$2.15 total cost.
- **Binary prompts**: \$42.88 (1896% increase).
- **Assembly prompts**: \$5.44 (153% increase).

![Token and cost comparison](comprehensive_analysis_results.png)

# Discussion

The results decisively show that binary formats, though compact on disk, are highly inefficient for LLM tokenizers. GPT-style tokenizers segment hexadecimal binary strings into many short subword tokens, inflating total token count and drastically increasing inference cost.

This contradicts the assumption that binary representations are "cheaper" or "more efficient." In contrast, source code retains more structure, allowing tokenizers to compress semantic information more effectively.

Assembly offers a middle ground—less efficient than source, but far superior to binary.

This study builds on our previous work on prompt diversity and model sampling strategies [@uniqueness2025].

# Implications

- Avoid binary prompts in production LLM applications.
- Source and assembly formats are preferable for both interpretability and efficiency.
- Future optimizations must balance cost with model performance and usability.

# Conclusion

Naive binary prompts catastrophically increase token count and cost when used with LLMs. These findings highlight the need for careful prompt design and suggest future research directions focused on tokenizer-aware representation strategies and downstream model accuracy.

# Acknowledgements

Thanks to the open-source and AI research communities for tools and datasets that enabled this work.

# References

::: {#refs}
:::
