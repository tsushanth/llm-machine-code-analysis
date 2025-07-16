#!/usr/bin/env python3
"""
Extended Configuration and Sample Generator
===========================================

This module provides configuration settings and extended sample generation
for the LLM token cost analysis project.
"""

import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    """Configuration for the token analysis."""
    model_name: str = "gpt-4"
    output_dir: str = "results"
    compiler_flags: List[str] = None
    optimization_levels: List[str] = None
    
    def __post_init__(self):
        if self.compiler_flags is None:
            self.compiler_flags = ["-O0", "-O1", "-O2", "-O3"]
        if self.optimization_levels is None:
            self.optimization_levels = ["none", "basic", "standard", "aggressive"]

class ExtendedSampleGenerator:
    """Generates more comprehensive code samples for analysis."""
    
    @staticmethod
    def get_algorithmic_samples() -> List[Tuple[str, str]]:
        """Return algorithmic code samples."""
        return [
            ("binary_search", """
#include <stdio.h>
int binary_search(int arr[], int l, int r, int x) {
    if (r >= l) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == x) return mid;
        if (arr[mid] > x) return binary_search(arr, l, mid - 1, x);
        return binary_search(arr, mid + 1, r, x);
    }
    return -1;
}
int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 10;
    int result = binary_search(arr, 0, n - 1, x);
    printf("Element found at index %d\\n", result);
    return 0;
}
"""),
            ("quick_sort", """
#include <stdio.h>
void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
int main() {
    int arr[] = {10, 7, 8, 9, 1, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    quickSort(arr, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    return 0;
}
"""),
            ("matrix_multiplication", """
#include <stdio.h>
#define N 3
void multiply(int A[N][N], int B[N][N], int C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
int main() {
    int A[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[N][N];
    multiply(A, B, C);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\\n");
    }
    return 0;
}
"""),
            ("hash_table_simple", """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TABLE_SIZE 10
typedef struct Node {
    char* key;
    int value;
    struct Node* next;
} Node;
Node* table[TABLE_SIZE];
int hash(char* key) {
    int sum = 0;
    for (int i = 0; key[i]; i++) {
        sum += key[i];
    }
    return sum % TABLE_SIZE;
}
void insert(char* key, int value) {
    int index = hash(key);
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->key = strdup(key);
    newNode->value = value;
    newNode->next = table[index];
    table[index] = newNode;
}
int search(char* key) {
    int index = hash(key);
    Node* current = table[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }
    return -1;
}
int main() {
    insert("apple", 5);
    insert("banana", 3);
    insert("orange", 8);
    printf("apple: %d\\n", search("apple"));
    printf("banana: %d\\n", search("banana"));
    return 0;
}
""")
        ]
    
    @staticmethod
    def get_data_structure_samples() -> List[Tuple[str, str]]:
        """Return data structure implementation samples."""
        return [
            ("stack", """
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 100
typedef struct {
    int items[MAX_SIZE];
    int top;
} Stack;
void push(Stack* s, int value) {
    if (s->top < MAX_SIZE - 1) {
        s->items[++s->top] = value;
    }
}
int pop(Stack* s) {
    if (s->top >= 0) {
        return s->items[s->top--];
    }
    return -1;
}
int main() {
    Stack s = {.top = -1};
    push(&s, 10);
    push(&s, 20);
    push(&s, 30);
    printf("%d\\n", pop(&s));
    printf("%d\\n", pop(&s));
    return 0;
}
"""),
            ("queue", """
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 100
typedef struct {
    int items[MAX_SIZE];
    int front, rear;
} Queue;
void enqueue(Queue* q, int value) {
    if (q->rear < MAX_SIZE - 1) {
        q->items[++q->rear] = value;
    }
}
int dequeue(Queue* q) {
    if (q->front <= q->rear) {
        return q->items[q->front++];
    }
    return -1;
}
int main() {
    Queue q = {.front = 0, .rear = -1};
    enqueue(&q, 10);
    enqueue(&q, 20);
    enqueue(&q, 30);
    printf("%d\\n", dequeue(&q));
    printf("%d\\n", dequeue(&q));
    return 0;
}
"""),
            ("binary_tree", """
#include <stdio.h>
#include <stdlib.h>
typedef struct Node {
    int data;
    struct Node* left;
    struct Node* right;
} Node;
Node* createNode(int data) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    node->left = node->right = NULL;
    return node;
}
void inorder(Node* root) {
    if (root) {
        inorder(root->left);
        printf("%d ", root->data);
        inorder(root->right);
    }
}
int main() {
    Node* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);
    inorder(root);
    return 0;
}
""")
        ]
    
    @staticmethod
    def get_mathematical_samples() -> List[Tuple[str, str]]:
        """Return mathematical computation samples."""
        return [
            ("prime_sieve", """
#include <stdio.h>
#include <stdbool.h>
void sieve(int n) {
    bool prime[n + 1];
    for (int i = 0; i <= n; i++) prime[i] = true;
    prime[0] = prime[1] = false;
    for (int p = 2; p * p <= n; p++) {
        if (prime[p]) {
            for (int i = p * p; i <= n; i += p) {
                prime[i] = false;
            }
        }
    }
    for (int i = 2; i <= n; i++) {
        if (prime[i]) printf("%d ", i);
    }
}
int main() {
    sieve(30);
    return 0;
}
"""),
            ("gcd_euclidean", """
#include <stdio.h>
int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}
int main() {
    int a = 48, b = 18;
    printf("GCD of %d and %d is %d\\n", a, b, gcd(a, b));
    return 0;
}
"""),
            ("factorial_iterative", """
#include <stdio.h>
long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
int main() {
    int n = 10;
    printf("Factorial of %d is %lld\\n", n, factorial(n));
    return 0;
}
""")
        ]
    
    @staticmethod
    def get_all_extended_samples() -> List[Tuple[str, str]]:
        """Return all extended samples combined."""
        generator = ExtendedSampleGenerator()
        all_samples = []
        all_samples.extend(generator.get_algorithmic_samples())
        all_samples.extend(generator.get_data_structure_samples())
        all_samples.extend(generator.get_mathematical_samples())
        return all_samples

class PromptTemplateGenerator:
    """Generates realistic prompts that would be sent to LLMs."""
    
    @staticmethod
    def create_code_explanation_prompt(code: str) -> str:
        """Create a prompt asking for code explanation."""
        return f"""Please explain how this C code works:

```c
{code}
```

Break down the algorithm step by step and explain the time complexity."""
    
    @staticmethod
    def create_code_optimization_prompt(code: str) -> str:
        """Create a prompt asking for code optimization."""
        return f"""Can you optimize this C code for better performance?

```c
{code}
```

Suggest improvements and explain why they would be more efficient."""
    
    @staticmethod
    def create_code_debugging_prompt(code: str) -> str:
        """Create a prompt asking for debugging help."""
        return f"""I'm getting unexpected results from this C code. Can you help me debug it?

```c
{code}
```

What might be causing issues and how can I fix them?"""
    
    @staticmethod
    def create_code_conversion_prompt(code: str) -> str:
        """Create a prompt asking for code conversion."""
        return f"""Convert this C code to Python:

```c
{code}
```

Maintain the same functionality but use Python idioms."""

class CostCalculator:
    """Calculates actual costs based on different pricing models."""
    
    def __init__(self):
        # Sample pricing (tokens per dollar) - update with actual rates
        self.pricing_models = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3": {"input": 0.015, "output": 0.075}
        }
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for given token counts."""
        if model not in self.pricing_models:
            return 0.0
        
        rates = self.pricing_models[model]
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        return input_cost + output_cost
    
    def calculate_savings(self, source_tokens: int, machine_tokens: int, 
                         output_tokens: int, model: str) -> Dict:
        """Calculate cost savings for machine code vs source code."""
        source_cost = self.calculate_cost(source_tokens, output_tokens, model)
        machine_cost = self.calculate_cost(machine_tokens, output_tokens, model)
        
        absolute_savings = source_cost - machine_cost
        percentage_savings = (absolute_savings / source_cost * 100) if source_cost > 0 else 0
        
        return {
            "source_cost": source_cost,
            "machine_cost": machine_cost,
            "absolute_savings": absolute_savings,
            "percentage_savings": percentage_savings
        }

def save_config(config: AnalysisConfig, filename: str = "analysis_config.json"):
    """Save configuration to JSON file."""
    with open(filename, 'w') as f:
        json.dump({
            "model_name": config.model_name,
            "output_dir": config.output_dir,
            "compiler_flags": config.compiler_flags,
            "optimization_levels": config.optimization_levels
        }, f, indent=2)

def load_config(filename: str = "analysis_config.json") -> AnalysisConfig:
    """Load configuration from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return AnalysisConfig(**data)
    except FileNotFoundError:
        return AnalysisConfig()  # Return default config

# Example usage and testing
if __name__ == "__main__":
    # Test the extended sample generator
    generator = ExtendedSampleGenerator()
    samples = generator.get_all_extended_samples()
    
    print(f"Generated {len(samples)} extended code samples:")
    for name, _ in samples[:3]:  # Show first 3
        print(f"- {name}")
    
    # Test prompt generation
    prompt_gen = PromptTemplateGenerator()
    sample_code = samples[0][1]  # Get first sample
    
    explanation_prompt = prompt_gen.create_code_explanation_prompt(sample_code)
    print(f"\nSample explanation prompt length: {len(explanation_prompt)} characters")
    
    # Test cost calculation
    calculator = CostCalculator()
    savings = calculator.calculate_savings(1000, 500, 200, "gpt-4")
    print(f"\nSample cost analysis:")
    print(f"Source cost: ${savings['source_cost']:.4f}")
    print(f"Machine cost: ${savings['machine_cost']:.4f}")
    print(f"Savings: ${savings['absolute_savings']:.4f} ({savings['percentage_savings']:.1f}%)")
    
    # Save sample configuration
    config = AnalysisConfig(model_name="gpt-4", output_dir="results")
    save_config(config)
    print(f"\nConfiguration saved to analysis_config.json")