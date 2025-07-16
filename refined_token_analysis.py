#!/usr/bin/env python3
"""
Refined Token Analysis: Pure Instruction Extraction
===================================================

This refined analysis extracts only the essential instruction content,
removing all overhead to test the true efficiency of machine code representations.
"""

import os
import subprocess
import tempfile
import tiktoken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Tuple, Optional

class PureInstructionExtractor:
    """Extracts only pure instruction content from compiled code."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_pure_instructions(self, source_code: str) -> Tuple[str, str]:
        """
        Extract pure instruction content from source code.
        Returns: (pure_opcodes, pure_assembly)
        """
        try:
            # Create temporary source file
            source_file = os.path.join(self.temp_dir, "temp.c")
            binary_file = os.path.join(self.temp_dir, "temp.bin")
            
            with open(source_file, 'w') as f:
                f.write(source_code)
            
            # Compile with minimal overhead
            compile_cmd = [
                "gcc", "-O2", "-nostdlib", "-nostartfiles",
                "-Wl,--build-id=none", "-Wl,--strip-all",
                "-o", binary_file, source_file
            ]
            
            # Try with minimal flags first, fallback to standard compilation
            try:
                result = subprocess.run(compile_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    # Fallback to standard compilation
                    compile_cmd = ["gcc", "-O2", "-o", binary_file, source_file]
                    result = subprocess.run(compile_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        return None, None
            except:
                return None, None
            
            # Extract .text section only using objdump
            objdump_cmd = ["objdump", "-d", "-j", ".text", binary_file]
            objdump_result = subprocess.run(objdump_cmd, capture_output=True, text=True)
            
            if objdump_result.returncode != 0:
                return None, None
            
            # Parse objdump output to extract pure instructions
            pure_opcodes, pure_assembly = self._parse_objdump_output(objdump_result.stdout)
            
            return pure_opcodes, pure_assembly
            
        except Exception as e:
            print(f"Error extracting instructions: {e}")
            return None, None
    
    def _parse_objdump_output(self, objdump_output: str) -> Tuple[str, str]:
        """Parse objdump output to extract pure instruction content."""
        lines = objdump_output.split('\n')
        opcodes = []
        assembly_instructions = []
        
        for line in lines:
            # Look for instruction lines (contain hex bytes and mnemonics)
            # Format: "  401000:	48 83 ec 08          	sub    $0x8,%rsp"
            if ':' in line and '\t' in line:
                try:
                    # Split by colon to separate address from instruction
                    parts = line.split(':', 1)
                    if len(parts) < 2:
                        continue
                    
                    instruction_part = parts[1].strip()
                    
                    # Split by tabs to separate hex bytes from assembly
                    tab_parts = instruction_part.split('\t')
                    if len(tab_parts) < 2:
                        continue
                    
                    # Extract hex bytes (opcodes)
                    hex_part = tab_parts[0].strip()
                    if hex_part:
                        # Remove spaces and keep only hex digits
                        hex_bytes = re.sub(r'[^0-9a-fA-F]', '', hex_part)
                        if hex_bytes:
                            opcodes.append(hex_bytes)
                    
                    # Extract assembly instruction (mnemonic + operands)
                    asm_part = tab_parts[-1].strip()
                    if asm_part:
                        assembly_instructions.append(asm_part)
                        
                except Exception:
                    continue
        
        # Join opcodes and assembly instructions
        pure_opcodes = ''.join(opcodes)
        pure_assembly = '\n'.join(assembly_instructions)
        
        return pure_opcodes, pure_assembly
    
    def __del__(self):
        """Cleanup temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class RefinedTokenAnalyzer:
    """Refined analyzer using pure instruction extraction."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
        self.extractor = PureInstructionExtractor()
    
    def analyze_refined_efficiency(self, samples: List[Tuple[str, str, int]]) -> pd.DataFrame:
        """Analyze token efficiency with pure instruction extraction."""
        
        results = []
        
        for name, source_code, estimated_lines in samples:
            print(f"Analyzing {name} (~{estimated_lines} lines)...")
            
            # Count source tokens
            source_tokens = len(self.tokenizer.encode(source_code))
            
            # Extract pure instructions
            pure_opcodes, pure_assembly = self.extractor.extract_pure_instructions(source_code)
            
            if pure_opcodes is None or pure_assembly is None:
                print(f"  Warning: Could not extract instructions for {name}")
                continue
            
            # Count tokens for pure representations
            pure_opcode_tokens = len(self.tokenizer.encode(pure_opcodes))
            pure_assembly_tokens = len(self.tokenizer.encode(pure_assembly))
            
            # Calculate efficiency
            opcode_efficiency = (source_tokens - pure_opcode_tokens) / source_tokens * 100
            assembly_efficiency = (source_tokens - pure_assembly_tokens) / source_tokens * 100
            
            # Calculate compression ratios
            opcode_compression = len(pure_opcodes) / len(source_code) if source_code else 0
            assembly_compression = len(pure_assembly) / len(source_code) if source_code else 0
            
            results.append({
                'name': name,
                'estimated_lines': estimated_lines,
                'source_tokens': source_tokens,
                'source_chars': len(source_code),
                'pure_opcode_tokens': pure_opcode_tokens,
                'pure_assembly_tokens': pure_assembly_tokens,
                'opcode_chars': len(pure_opcodes),
                'assembly_chars': len(pure_assembly),
                'opcode_efficiency_pct': opcode_efficiency,
                'assembly_efficiency_pct': assembly_efficiency,
                'opcode_compression_ratio': opcode_compression,
                'assembly_compression_ratio': assembly_compression,
                'tokens_per_line': source_tokens / estimated_lines,
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
    
    def create_refined_visualizations(self, df: pd.DataFrame):
        """Create visualizations for refined analysis."""
        
        if df.empty:
            print("No data to visualize.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Pure efficiency vs program size
        ax1 = axes[0, 0]
        ax1.scatter(df['estimated_lines'], df['opcode_efficiency_pct'], 
                   label='Pure Opcodes', alpha=0.7, s=100, color='blue')
        ax1.scatter(df['estimated_lines'], df['assembly_efficiency_pct'], 
                   label='Pure Assembly', alpha=0.7, s=100, color='red')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Program Size (Lines of Code)')
        ax1.set_ylabel('Token Efficiency (%)')
        ax1.set_title('Pure Instruction Token Efficiency vs Program Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add trend lines
        if len(df) > 1:
            z1 = np.polyfit(df['estimated_lines'], df['opcode_efficiency_pct'], 1)
            p1 = np.poly1d(z1)
            ax1.plot(df['estimated_lines'], p1(df['estimated_lines']), 
                    "blue", linestyle='--', alpha=0.8)
            
            z2 = np.polyfit(df['estimated_lines'], df['assembly_efficiency_pct'], 1)
            p2 = np.poly1d(z2)
            ax1.plot(df['estimated_lines'], p2(df['estimated_lines']), 
                    "red", linestyle='--', alpha=0.8)
        
        # 2. Compression ratios
        ax2 = axes[0, 1]
        ax2.scatter(df['estimated_lines'], df['opcode_compression_ratio'], 
                   label='Opcode Compression', alpha=0.7, s=100, color='green')
        ax2.scatter(df['estimated_lines'], df['assembly_compression_ratio'], 
                   label='Assembly Compression', alpha=0.7, s=100, color='orange')
        ax2.set_xlabel('Program Size (Lines of Code)')
        ax2.set_ylabel('Compression Ratio (machine/source)')
        ax2.set_title('Character Compression Ratios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Token counts comparison
        ax3 = axes[1, 0]
        categories = df['scale_category'].unique()
        if len(categories) > 0:
            x_pos = np.arange(len(categories))
            
            source_means = [df[df['scale_category'] == cat]['source_tokens'].mean() 
                           for cat in categories]
            opcode_means = [df[df['scale_category'] == cat]['pure_opcode_tokens'].mean() 
                           for cat in categories]
            assembly_means = [df[df['scale_category'] == cat]['pure_assembly_tokens'].mean() 
                             for cat in categories]
            
            width = 0.25
            ax3.bar(x_pos - width, source_means, width, label='Source', alpha=0.8)
            ax3.bar(x_pos, opcode_means, width, label='Pure Opcodes', alpha=0.8)
            ax3.bar(x_pos + width, assembly_means, width, label='Pure Assembly', alpha=0.8)
            
            ax3.set_xlabel('Scale Category')
            ax3.set_ylabel('Average Token Count')
            ax3.set_title('Token Counts by Scale Category (Pure Instructions)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(categories)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency improvement over scale
        ax4 = axes[1, 1]
        df_sorted = df.sort_values('estimated_lines')
        
        ax4.plot(df_sorted['estimated_lines'], df_sorted['opcode_efficiency_pct'], 
                'o-', label='Opcode efficiency', alpha=0.8, color='blue')
        ax4.plot(df_sorted['estimated_lines'], df_sorted['assembly_efficiency_pct'], 
                's-', label='Assembly efficiency', alpha=0.8, color='red')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Break-even')
        
        ax4.set_xlabel('Program Size (Lines of Code)')
        ax4.set_ylabel('Token Efficiency (%)')
        ax4.set_title('Pure Instruction Efficiency Trends')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('refined_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed findings
        self._print_detailed_findings(df)
    
    def _print_detailed_findings(self, df: pd.DataFrame):
        """Print detailed analysis findings."""
        
        print("\n" + "="*60)
        print("REFINED ANALYSIS RESULTS")
        print("="*60)
        
        if df.empty:
            print("No successful analyses to report.")
            return
        
        # Overall statistics
        print(f"Successfully analyzed: {len(df)} samples")
        print(f"Average source tokens: {df['source_tokens'].mean():.1f}")
        print(f"Average opcode tokens: {df['pure_opcode_tokens'].mean():.1f}")
        print(f"Average assembly tokens: {df['pure_assembly_tokens'].mean():.1f}")
        
        # Efficiency statistics
        print(f"\nEfficiency Statistics:")
        print(f"Opcode efficiency: {df['opcode_efficiency_pct'].mean():.1f}% average")
        print(f"Assembly efficiency: {df['assembly_efficiency_pct'].mean():.1f}% average")
        
        # Best cases
        best_opcode = df.loc[df['opcode_efficiency_pct'].idxmax()]
        best_assembly = df.loc[df['assembly_efficiency_pct'].idxmax()]
        
        print(f"\nBest efficiency achieved:")
        print(f"Opcode: {best_opcode['opcode_efficiency_pct']:.1f}% ({best_opcode['name']})")
        print(f"Assembly: {best_assembly['assembly_efficiency_pct']:.1f}% ({best_assembly['name']})")
        
        # Break-even analysis
        positive_opcode = df[df['opcode_efficiency_pct'] > 0]
        positive_assembly = df[df['assembly_efficiency_pct'] > 0]
        
        if not positive_opcode.empty:
            min_opcode_size = positive_opcode['estimated_lines'].min()
            print(f"\nOpcode becomes efficient at: {min_opcode_size} lines")
        else:
            print(f"\nOpcode efficiency range: {df['opcode_efficiency_pct'].min():.1f}% to {df['opcode_efficiency_pct'].max():.1f}%")
        
        if not positive_assembly.empty:
            min_assembly_size = positive_assembly['estimated_lines'].min()
            print(f"Assembly becomes efficient at: {min_assembly_size} lines")
        else:
            print(f"Assembly efficiency range: {df['assembly_efficiency_pct'].min():.1f}% to {df['assembly_efficiency_pct'].max():.1f}%")
        
        # Correlation analysis
        if len(df) > 1:
            opcode_corr = df['estimated_lines'].corr(df['opcode_efficiency_pct'])
            assembly_corr = df['estimated_lines'].corr(df['assembly_efficiency_pct'])
            
            print(f"\nCorrelation with program size:")
            print(f"Opcode efficiency: {opcode_corr:.3f}")
            print(f"Assembly efficiency: {assembly_corr:.3f}")

class LargeCodeSampleGenerator:
    """Generates progressively larger code samples for scale testing."""
    
    @staticmethod
    def get_all_samples() -> List[Tuple[str, str, int]]:
        """Get all samples sorted by size."""
        return [
            ("basic_calculator", '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double calculate(double a, double b, char op) {
    switch(op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return b != 0 ? a / b : 0;
        case '^': return pow(a, b);
        default: return 0;
    }
}

int main() {
    double a, b, result;
    char op;
    
    printf("Calculator\\n");
    printf("Enter first number: ");
    scanf("%lf", &a);
    printf("Enter operator: ");
    scanf(" %c", &op);
    printf("Enter second number: ");
    scanf("%lf", &b);
    
    result = calculate(a, b, op);
    printf("Result: %.2f\\n", result);
    
    return 0;
}
''', 50),
            
            ("simple_data_structures", '''
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

typedef struct Node {
    int data;
    struct Node* next;
} Node;

Node* createNode(int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

void insertAtBeginning(Node** head, int data) {
    Node* newNode = createNode(data);
    newNode->next = *head;
    *head = newNode;
}

void printList(Node* head) {
    Node* current = head;
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\\n");
}

int main() {
    Stack stack = {.top = -1};
    push(&stack, 10);
    push(&stack, 20);
    printf("Popped: %d\\n", pop(&stack));
    
    Node* head = NULL;
    insertAtBeginning(&head, 1);
    insertAtBeginning(&head, 2);
    insertAtBeginning(&head, 3);
    printList(head);
    
    return 0;
}
''', 120),
            
            ("sorting_algorithms", '''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void bubbleSort(int arr[], int n) {
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

void selectionSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        int minIdx = i;
        for (int j = i+1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        int temp = arr[minIdx];
        arr[minIdx] = arr[i];
        arr[i] = temp;
    }
}

void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int* leftArr = (int*)malloc(n1 * sizeof(int));
    int* rightArr = (int*)malloc(n2 * sizeof(int));
    
    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
    
    free(leftArr);
    free(rightArr);
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\\n");
}

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr)/sizeof(arr[0]);
    
    printf("Original array: ");
    printArray(arr, n);
    
    bubbleSort(arr, n);
    printf("Sorted array: ");
    printArray(arr, n);
    
    return 0;
}
''', 250),
            
            ("binary_search_tree", '''
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* left;
    struct Node* right;
} Node;

Node* createNode(int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

Node* insert(Node* root, int data) {
    if (root == NULL) {
        return createNode(data);
    }
    
    if (data < root->data) {
        root->left = insert(root->left, data);
    } else if (data > root->data) {
        root->right = insert(root->right, data);
    }
    
    return root;
}

Node* search(Node* root, int data) {
    if (root == NULL || root->data == data) {
        return root;
    }
    
    if (data < root->data) {
        return search(root->left, data);
    }
    
    return search(root->right, data);
}

Node* findMin(Node* root) {
    while (root->left != NULL) {
        root = root->left;
    }
    return root;
}

Node* deleteNode(Node* root, int data) {
    if (root == NULL) {
        return root;
    }
    
    if (data < root->data) {
        root->left = deleteNode(root->left, data);
    } else if (data > root->data) {
        root->right = deleteNode(root->right, data);
    } else {
        if (root->left == NULL) {
            Node* temp = root->right;
            free(root);
            return temp;
        } else if (root->right == NULL) {
            Node* temp = root->left;
            free(root);
            return temp;
        }
        
        Node* temp = findMin(root->right);
        root->data = temp->data;
        root->right = deleteNode(root->right, temp->data);
    }
    
    return root;
}

void inorder(Node* root) {
    if (root != NULL) {
        inorder(root->left);
        printf("%d ", root->data);
        inorder(root->right);
    }
}

void preorder(Node* root) {
    if (root != NULL) {
        printf("%d ", root->data);
        preorder(root->left);
        preorder(root->right);
    }
}

void postorder(Node* root) {
    if (root != NULL) {
        postorder(root->left);
        postorder(root->right);
        printf("%d ", root->data);
    }
}

int main() {
    Node* root = NULL;
    
    root = insert(root, 50);
    root = insert(root, 30);
    root = insert(root, 20);
    root = insert(root, 40);
    root = insert(root, 70);
    root = insert(root, 60);
    root = insert(root, 80);
    
    printf("Inorder traversal: ");
    inorder(root);
    printf("\\n");
    
    printf("Preorder traversal: ");
    preorder(root);
    printf("\\n");
    
    printf("Postorder traversal: ");
    postorder(root);
    printf("\\n");
    
    Node* found = search(root, 40);
    if (found != NULL) {
        printf("Found %d\\n", found->data);
    } else {
        printf("Not found\\n");
    }
    
    root = deleteNode(root, 20);
    printf("After deleting 20: ");
    inorder(root);
    printf("\\n");
    
    return 0;
}
''', 300),
            
            ("matrix_operations", '''
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 100

void printMatrix(int matrix[][MAX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\\n");
    }
}

void addMatrices(int a[][MAX_SIZE], int b[][MAX_SIZE], int result[][MAX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

void multiplyMatrices(int a[][MAX_SIZE], int b[][MAX_SIZE], int result[][MAX_SIZE], 
                     int rows1, int cols1, int rows2, int cols2) {
    if (cols1 != rows2) {
        printf("Matrix multiplication not possible\\n");
        return;
    }
    
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void transposeMatrix(int matrix[][MAX_SIZE], int result[][MAX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
}

int determinant(int matrix[][MAX_SIZE], int n) {
    if (n == 1) {
        return matrix[0][0];
    }
    
    if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    
    int det = 0;
    int cofactor[MAX_SIZE][MAX_SIZE];
    
    for (int f = 0; f < n; f++) {
        int c = 0, r = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j != f) {
                    cofactor[r][c++] = matrix[i][j];
                }
            }
            r++;
            c = 0;
        }
        
        det += (f % 2 == 0 ? 1 : -1) * matrix[0][f] * determinant(cofactor, n - 1);
    }
    
    return det;
}

int main() {
    int a[MAX_SIZE][MAX_SIZE] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int b[MAX_SIZE][MAX_SIZE] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int result[MAX_SIZE][MAX_SIZE];
    
    printf("Matrix A:\\n");
    printMatrix(a, 3, 3);
    
    printf("Matrix B:\\n");
    printMatrix(b, 3, 3);
    
    addMatrices(a, b, result, 3, 3);
    printf("A + B:\\n");
    printMatrix(result, 3, 3);
    
    multiplyMatrices(a, b, result, 3, 3, 3, 3);
    printf("A * B:\\n");
    printMatrix(result, 3, 3);
    
    transposeMatrix(a, result, 3, 3);
    printf("Transpose of A:\\n");
    printMatrix(result, 3, 3);
    
    printf("Determinant of A: %d\\n", determinant(a, 3));
    
    return 0;
}
''', 400),
            
            ("hash_table_complete", '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 101

typedef struct HashNode {
    char* key;
    int value;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode* buckets[TABLE_SIZE];
    int size;
} HashTable;

unsigned int hash(const char* key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

HashTable* createHashTable() {
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    for (int i = 0; i < TABLE_SIZE; i++) {
        table->buckets[i] = NULL;
    }
    table->size = 0;
    return table;
}

void insert(HashTable* table, const char* key, int value) {
    unsigned int index = hash(key);
    HashNode* newNode = (HashNode*)malloc(sizeof(HashNode));
    newNode->key = strdup(key);
    newNode->value = value;
    newNode->next = table->buckets[index];
    table->buckets[index] = newNode;
    table->size++;
}

int search(HashTable* table, const char* key) {
    unsigned int index = hash(key);
    HashNode* current = table->buckets[index];
    
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }
    
    return -1;
}

void delete(HashTable* table, const char* key) {
    unsigned int index = hash(key);
    HashNode* current = table->buckets[index];
    HashNode* prev = NULL;
    
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            if (prev == NULL) {
                table->buckets[index] = current->next;
            } else {
                prev->next = current->next;
            }
            free(current->key);
            free(current);
            table->size--;
            return;
        }
        prev = current;
        current = current->next;
    }
}

void printHashTable(HashTable* table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (table->buckets[i] != NULL) {
            printf("Bucket %d: ", i);
            HashNode* current = table->buckets[i];
            while (current != NULL) {
                printf("(%s: %d) ", current->key, current->value);
                current = current->next;
            }
            printf("\\n");
        }
    }
}

int main() {
    HashTable* table = createHashTable();
    
    insert(table, "apple", 5);
    insert(table, "banana", 3);
    insert(table, "orange", 8);
    insert(table, "grape", 12);
    insert(table, "watermelon", 20);
    
    printf("Hash table contents:\\n");
    printHashTable(table);
    
    printf("\\nSearch results:\\n");
    printf("apple: %d\\n", search(table, "apple"));
    printf("banana: %d\\n", search(table, "banana"));
    printf("kiwi: %d\\n", search(table, "kiwi"));
    
    delete(table, "banana");
    printf("\\nAfter deleting banana:\\n");
    printHashTable(table);
    
    return 0;
}
''', 500),
        ]

def main():
    """Run the refined analysis."""
    
    print("REFINED TOKEN EFFICIENCY ANALYSIS")
    print("=" * 40)
    print("Testing with pure instruction extraction...")
    print()
    
    # Create samples
    generator = LargeCodeSampleGenerator()
    samples = generator.get_all_samples()
    
    # Initialize refined analyzer
    analyzer = RefinedTokenAnalyzer("gpt-4")
    
    # Run refined analysis
    results_df = analyzer.analyze_refined_efficiency(samples)
    
    if not results_df.empty:
        # Display results
        print("\nRefined Results Summary:")
        print(results_df[['name', 'estimated_lines', 'source_tokens', 
                         'opcode_efficiency_pct', 'assembly_efficiency_pct']].to_string(index=False))
        
        # Create visualizations
        analyzer.create_refined_visualizations(results_df)
        
        # Save results
        results_df.to_csv('refined_analysis_results.csv', index=False)
        print(f"\nResults saved to 'refined_analysis_results.csv'")
    else:
        print("No successful analyses. Check compilation environment.")

if __name__ == "__main__":
    main()