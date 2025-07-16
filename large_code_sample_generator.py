#!/usr/bin/env python3
"""
Large Code Analysis: Testing Scale-Dependent Token Efficiency
============================================================

This analysis tests the hypothesis that machine code becomes more token-efficient
than source code as the size and complexity of programs increase.

The theory: Fixed overhead (headers, formatting) becomes proportionally smaller
as the actual code content grows larger.
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
import re

class LargeCodeSampleGenerator:
    """Generates progressively larger code samples for scale testing."""
    
    @staticmethod
    def get_small_samples() -> List[Tuple[str, str, int]]:
        """Return small code samples (50-200 lines)."""
        return [
            ("basic_calculator", """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double value;
    char operator;
} Operation;

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
    
    printf("Enhanced Calculator\\n");
    printf("Supported operations: +, -, *, /, ^\\n");
    
    printf("Enter first number: ");
    scanf("%lf", &a);
    
    printf("Enter operator: ");
    scanf(" %c", &op);
    
    printf("Enter second number: ");
    scanf("%lf", &b);
    
    result = calculate(a, b, op);
    printf("Result: %.2f %c %.2f = %.2f\\n", a, op, b, result);
    
    return 0;
}
""", 50),
            
            ("simple_data_structures", """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 100

typedef struct Node {
    int data;
    struct Node* next;
} Node;

typedef struct {
    int items[MAX_SIZE];
    int top;
} Stack;

typedef struct {
    int items[MAX_SIZE];
    int front, rear;
} Queue;

// Stack operations
void push(Stack* s, int value) {
    if (s->top < MAX_SIZE - 1) {
        s->items[++s->top] = value;
        printf("Pushed %d\\n", value);
    } else {
        printf("Stack overflow\\n");
    }
}

int pop(Stack* s) {
    if (s->top >= 0) {
        return s->items[s->top--];
    }
    printf("Stack underflow\\n");
    return -1;
}

// Queue operations
void enqueue(Queue* q, int value) {
    if (q->rear < MAX_SIZE - 1) {
        q->items[++q->rear] = value;
        printf("Enqueued %d\\n", value);
    } else {
        printf("Queue overflow\\n");
    }
}

int dequeue(Queue* q) {
    if (q->front <= q->rear) {
        return q->items[q->front++];
    }
    printf("Queue underflow\\n");
    return -1;
}

// Linked list operations
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
    // Test stack
    Stack stack = {.top = -1};
    push(&stack, 10);
    push(&stack, 20);
    push(&stack, 30);
    printf("Popped: %d\\n", pop(&stack));
    
    // Test queue
    Queue queue = {.front = 0, .rear = -1};
    enqueue(&queue, 100);
    enqueue(&queue, 200);
    enqueue(&queue, 300);
    printf("Dequeued: %d\\n", dequeue(&queue));
    
    // Test linked list
    Node* head = NULL;
    insertAtBeginning(&head, 1);
    insertAtBeginning(&head, 2);
    insertAtBeginning(&head, 3);
    printf("Linked list: ");
    printList(head);
    
    return 0;
}
""", 120),
        ]
    
    @staticmethod
    def get_medium_samples() -> List[Tuple[str, str, int]]:
        """Return medium code samples (200-500 lines)."""
        return [
            ("sorting_algorithms", """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_SIZE 1000

// Function prototypes
void bubbleSort(int arr[], int n);
void selectionSort(int arr[], int n);
void insertionSort(int arr[], int n);
void mergeSort(int arr[], int left, int right);
void quickSort(int arr[], int low, int high);
void heapSort(int arr[], int n);
void printArray(int arr[], int n);
void generateRandomArray(int arr[], int n);
int* copyArray(int arr[], int n);

// Bubble Sort
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

// Selection Sort
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

// Insertion Sort
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

// Merge function for merge sort
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

// Merge Sort
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Partition function for quick sort
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return (i + 1);
}

// Quick Sort
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Heap operations
void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && arr[left] > arr[largest])
        largest = left;
    
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
        heapify(arr, n, largest);
    }
}

// Heap Sort
void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    
    for (int i = n - 1; i > 0; i--) {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        heapify(arr, i, 0);
    }
}

// Utility functions
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\\n");
}

void generateRandomArray(int arr[], int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }
}

int* copyArray(int arr[], int n) {
    int* copy = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        copy[i] = arr[i];
    }
    return copy;
}

int main() {
    int n = 20;
    int originalArray[MAX_SIZE];
    
    printf("Sorting Algorithm Demonstration\\n");
    printf("===============================\\n");
    
    generateRandomArray(originalArray, n);
    printf("Original array: ");
    printArray(originalArray, n);
    
    // Test each sorting algorithm
    int* testArray;
    
    printf("\\nBubble Sort: ");
    testArray = copyArray(originalArray, n);
    bubbleSort(testArray, n);
    printArray(testArray, n);
    free(testArray);
    
    printf("Selection Sort: ");
    testArray = copyArray(originalArray, n);
    selectionSort(testArray, n);
    printArray(testArray, n);
    free(testArray);
    
    printf("Insertion Sort: ");
    testArray = copyArray(originalArray, n);
    insertionSort(testArray, n);
    printArray(testArray, n);
    free(testArray);
    
    printf("Merge Sort: ");
    testArray = copyArray(originalArray, n);
    mergeSort(testArray, 0, n-1);
    printArray(testArray, n);
    free(testArray);
    
    printf("Quick Sort: ");
    testArray = copyArray(originalArray, n);
    quickSort(testArray, 0, n-1);
    printArray(testArray, n);
    free(testArray);
    
    printf("Heap Sort: ");
    testArray = copyArray(originalArray, n);
    heapSort(testArray, n);
    printArray(testArray, n);
    free(testArray);
    
    return 0;
}
""", 250),
            
            ("hash_table_implementation", """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 101
#define MAX_KEY_LENGTH 100
#define MAX_VALUE_LENGTH 200

typedef struct HashNode {
    char* key;
    char* value;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode* buckets[TABLE_SIZE];
    int size;
    int collisions;
} HashTable;

// Hash function implementations
unsigned int hash_djb2(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

unsigned int hash_sdbm(const char* str) {
    unsigned int hash = 0;
    int c;
    while ((c = *str++)) {
        hash = c + (hash << 6) + (hash << 16) - hash;
    }
    return hash % TABLE_SIZE;
}

unsigned int hash_simple(const char* str) {
    unsigned int hash = 0;
    for (int i = 0; str[i]; i++) {
        hash += str[i];
    }
    return hash % TABLE_SIZE;
}

// Hash table operations
HashTable* createHashTable() {
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    for (int i = 0; i < TABLE_SIZE; i++) {
        table->buckets[i] = NULL;
    }
    table->size = 0;
    table->collisions = 0;
    return table;
}

HashNode* createNode(const char* key, const char* value) {
    HashNode* node = (HashNode*)malloc(sizeof(HashNode));
    node->key = (char*)malloc((strlen(key) + 1) * sizeof(char));
    node->value = (char*)malloc((strlen(value) + 1) * sizeof(char));
    strcpy(node->key, key);
    strcpy(node->value, value);
    node->next = NULL;
    return node;
}

void insert(HashTable* table, const char* key, const char* value) {
    unsigned int index = hash_djb2(key);
    HashNode* newNode = createNode(key, value);
    
    if (table->buckets[index] == NULL) {
        table->buckets[index] = newNode;
    } else {
        table->collisions++;
        HashNode* current = table->buckets[index];
        
        // Check if key already exists
        while (current != NULL) {
            if (strcmp(current->key, key) == 0) {
                // Update existing value
                free(current->value);
                current->value = (char*)malloc((strlen(value) + 1) * sizeof(char));
                strcpy(current->value, value);
                free(newNode->key);
                free(newNode->value);
                free(newNode);
                return;
            }
            if (current->next == NULL) break;
            current = current->next;
        }
        
        // Add new node at end of chain
        current->next = newNode;
    }
    table->size++;
}

char* search(HashTable* table, const char* key) {
    unsigned int index = hash_djb2(key);
    HashNode* current = table->buckets[index];
    
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }
    return NULL;
}

int delete(HashTable* table, const char* key) {
    unsigned int index = hash_djb2(key);
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
            free(current->value);
            free(current);
            table->size--;
            return 1;
        }
        prev = current;
        current = current->next;
    }
    return 0;
}

void printHashTable(HashTable* table) {
    printf("Hash Table Contents:\\n");
    printf("Size: %d, Collisions: %d\\n", table->size, table->collisions);
    printf("Load Factor: %.2f\\n", (double)table->size / TABLE_SIZE);
    
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (table->buckets[i] != NULL) {
            printf("Bucket %d: ", i);
            HashNode* current = table->buckets[i];
            while (current != NULL) {
                printf("(%s: %s)", current->key, current->value);
                if (current->next != NULL) printf(" -> ");
                current = current->next;
            }
            printf("\\n");
        }
    }
}

void freeHashTable(HashTable* table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        HashNode* current = table->buckets[i];
        while (current != NULL) {
            HashNode* temp = current;
            current = current->next;
            free(temp->key);
            free(temp->value);
            free(temp);
        }
    }
    free(table);
}

// Test data and functions
void runTests(HashTable* table) {
    printf("Running Hash Table Tests...\\n");
    
    // Insert test data
    insert(table, "apple", "A red or green fruit");
    insert(table, "banana", "A yellow curved fruit");
    insert(table, "orange", "An orange citrus fruit");
    insert(table, "grape", "Small round fruits in clusters");
    insert(table, "strawberry", "A red berry with seeds on outside");
    insert(table, "blueberry", "A small blue berry");
    insert(table, "watermelon", "A large green fruit with red flesh");
    insert(table, "pineapple", "A tropical fruit with spiky skin");
    insert(table, "mango", "A sweet tropical fruit");
    insert(table, "kiwi", "A brown fuzzy fruit with green flesh");
    
    // Search tests
    printf("\\nSearch Tests:\\n");
    char* result = search(table, "apple");
    printf("apple: %s\\n", result ? result : "Not found");
    
    result = search(table, "banana");
    printf("banana: %s\\n", result ? result : "Not found");
    
    result = search(table, "nonexistent");
    printf("nonexistent: %s\\n", result ? result : "Not found");
    
    // Update test
    insert(table, "apple", "A crunchy fruit that comes in many varieties");
    result = search(table, "apple");
    printf("Updated apple: %s\\n", result ? result : "Not found");
    
    // Delete test
    printf("\\nDelete Tests:\\n");
    printf("Deleting 'grape': %s\\n", delete(table, "grape") ? "Success" : "Failed");
    printf("Deleting 'nonexistent': %s\\n", delete(table, "nonexistent") ? "Success" : "Failed");
    
    printHashTable(table);
}

int main() {
    printf("Advanced Hash Table Implementation\\n");
    printf("=================================\\n");
    
    HashTable* table = createHashTable();
    runTests(table);
    
    freeHashTable(table);
    return 0;
}
""", 300),
        ]
    
    @staticmethod 
    def get_large_samples() -> List[Tuple[str, str, int]]:
        """Return large code samples (500-1500 lines)."""
        return [
            ("basic_compiler", """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LENGTH 100
#define MAX_TOKENS 1000
#define MAX_VARIABLES 100

// Token types
typedef enum {
    TOKEN_NUMBER,
    TOKEN_IDENTIFIER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_ASSIGN,
    TOKEN_SEMICOLON,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_FOR,
    TOKEN_INT,
    TOKEN_PRINT,
    TOKEN_EOF,
    TOKEN_UNKNOWN
} TokenType;

typedef struct {
    TokenType type;
    char value[MAX_TOKEN_LENGTH];
    int line;
    int column;
} Token;

typedef struct {
    char name[MAX_TOKEN_LENGTH];
    int value;
    int defined;
} Variable;

// AST Node types
typedef enum {
    NODE_NUMBER,
    NODE_IDENTIFIER,
    NODE_BINARY_OP,
    NODE_ASSIGN,
    NODE_PRINT,
    NODE_IF,
    NODE_WHILE,
    NODE_BLOCK,
    NODE_DECLARATION
} NodeType;

typedef struct ASTNode {
    NodeType type;
    union {
        int number;
        char identifier[MAX_TOKEN_LENGTH];
        struct {
            struct ASTNode* left;
            struct ASTNode* right;
            TokenType operator;
        } binary_op;
        struct {
            char variable[MAX_TOKEN_LENGTH];
            struct ASTNode* expression;
        } assignment;
        struct {
            struct ASTNode* expression;
        } print_stmt;
        struct {
            struct ASTNode* condition;
            struct ASTNode* then_stmt;
            struct ASTNode* else_stmt;
        } if_stmt;
        struct {
            struct ASTNode* condition;
            struct ASTNode* body;
        } while_stmt;
        struct {
            struct ASTNode** statements;
            int count;
        } block;
        struct {
            char variable[MAX_TOKEN_LENGTH];
            struct ASTNode* initializer;
        } declaration;
    };
} ASTNode;

// Global variables
Token tokens[MAX_TOKENS];
int token_count = 0;
int current_token = 0;
Variable variables[MAX_VARIABLES];
int variable_count = 0;

// Forward declarations
ASTNode* parseExpression();
ASTNode* parseStatement();
ASTNode* parseBlock();
int evaluateExpression(ASTNode* node);
void executeStatement(ASTNode* node);

// Lexer functions
int isKeyword(const char* str) {
    return (strcmp(str, "if") == 0 || strcmp(str, "else") == 0 || 
            strcmp(str, "while") == 0 || strcmp(str, "for") == 0 ||
            strcmp(str, "int") == 0 || strcmp(str, "print") == 0);
}

TokenType getKeywordType(const char* str) {
    if (strcmp(str, "if") == 0) return TOKEN_IF;
    if (strcmp(str, "else") == 0) return TOKEN_ELSE;
    if (strcmp(str, "while") == 0) return TOKEN_WHILE;
    if (strcmp(str, "for") == 0) return TOKEN_FOR;
    if (strcmp(str, "int") == 0) return TOKEN_INT;
    if (strcmp(str, "print") == 0) return TOKEN_PRINT;
    return TOKEN_UNKNOWN;
}

void addToken(TokenType type, const char* value, int line, int column) {
    if (token_count < MAX_TOKENS) {
        tokens[token_count].type = type;
        strcpy(tokens[token_count].value, value);
        tokens[token_count].line = line;
        tokens[token_count].column = column;
        token_count++;
    }
}

void tokenize(const char* source) {
    int line = 1, column = 1;
    int i = 0;
    int length = strlen(source);
    
    while (i < length) {
        char c = source[i];
        
        // Skip whitespace
        if (isspace(c)) {
            if (c == '\\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
            i++;
            continue;
        }
        
        // Numbers
        if (isdigit(c)) {
            char number[MAX_TOKEN_LENGTH];
            int j = 0;
            while (i < length && isdigit(source[i])) {
                number[j++] = source[i++];
            }
            number[j] = '\\0';
            addToken(TOKEN_NUMBER, number, line, column);
            column += j;
            continue;
        }
        
        // Identifiers and keywords
        if (isalpha(c) || c == '_') {
            char identifier[MAX_TOKEN_LENGTH];
            int j = 0;
            while (i < length && (isalnum(source[i]) || source[i] == '_')) {
                identifier[j++] = source[i++];
            }
            identifier[j] = '\\0';
            
            TokenType type = isKeyword(identifier) ? getKeywordType(identifier) : TOKEN_IDENTIFIER;
            addToken(type, identifier, line, column);
            column += j;
            continue;
        }
        
        // Single character tokens
        switch (c) {
            case '+': addToken(TOKEN_PLUS, "+", line, column); break;
            case '-': addToken(TOKEN_MINUS, "-", line, column); break;
            case '*': addToken(TOKEN_MULTIPLY, "*", line, column); break;
            case '/': addToken(TOKEN_DIVIDE, "/", line, column); break;
            case '=': addToken(TOKEN_ASSIGN, "=", line, column); break;
            case ';': addToken(TOKEN_SEMICOLON, ";", line, column); break;
            case '(': addToken(TOKEN_LPAREN, "(", line, column); break;
            case ')': addToken(TOKEN_RPAREN, ")", line, column); break;
            case '{': addToken(TOKEN_LBRACE, "{", line, column); break;
            case '}': addToken(TOKEN_RBRACE, "}", line, column); break;
            default:
                printf("Unknown character: %c at line %d, column %d\\n", c, line, column);
                break;
        }
        i++;
        column++;
    }
    
    addToken(TOKEN_EOF, "", line, column);
}

// Parser functions
Token getCurrentToken() {
    if (current_token < token_count) {
        return tokens[current_token];
    }
    return tokens[token_count - 1]; // EOF token
}

void consumeToken() {
    if (current_token < token_count - 1) {
        current_token++;
    }
}

ASTNode* createNumberNode(int value) {
    ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
    node->type = NODE_NUMBER;
    node->number = value;
    return node;
}

ASTNode* createIdentifierNode(const char* name) {
    ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
    node->type = NODE_IDENTIFIER;
    strcpy(node->identifier, name);
    return node;
}

ASTNode* createBinaryOpNode(ASTNode* left, TokenType op, ASTNode* right) {
    ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
    node->type = NODE_BINARY_OP;
    node->binary_op.left = left;
    node->binary_op.operator = op;
    node->binary_op.right = right;
    return node;
}

ASTNode* parsePrimary() {
    Token token = getCurrentToken();
    
    if (token.type == TOKEN_NUMBER) {
        consumeToken();
        return createNumberNode(atoi(token.value));
    } else if (token.type == TOKEN_IDENTIFIER) {
        consumeToken();
        return createIdentifierNode(token.value);
    } else if (token.type == TOKEN_LPAREN) {
        consumeToken();
        ASTNode* expr = parseExpression();
        if (getCurrentToken().type == TOKEN_RPAREN) {
            consumeToken();
        }
        return expr;
    }
    
    return NULL;
}

ASTNode* parseTerm() {
    ASTNode* left = parsePrimary();
    
    while (getCurrentToken().type == TOKEN_MULTIPLY || getCurrentToken().type == TOKEN_DIVIDE) {
        TokenType op = getCurrentToken().type;
        consumeToken();
        ASTNode* right = parsePrimary();
        left = createBinaryOpNode(left, op, right);
    }
    
    return left;
}

ASTNode* parseExpression() {
    ASTNode* left = parseTerm();
    
    while (getCurrentToken().type == TOKEN_PLUS || getCurrentToken().type == TOKEN_MINUS) {
        TokenType op = getCurrentToken().type;
        consumeToken();
        ASTNode* right = parseTerm();
        left = createBinaryOpNode(left, op, right);
    }
    
    return left;
}

ASTNode* parseAssignment() {
    Token token = getCurrentToken();
    
    if (token.type == TOKEN_IDENTIFIER) {
        char varName[MAX_TOKEN_LENGTH];
        strcpy(varName, token.value);
        consumeToken();
        
        if (getCurrentToken().type == TOKEN_ASSIGN) {
            consumeToken();
            ASTNode* expr = parseExpression();
            ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
            node->type = NODE_ASSIGN;
            strcpy(node->assignment.variable, varName);
            node->assignment.expression = expr;
            return node;
        }
    }
    
    return parseExpression();
}

ASTNode* parseStatement() {
    Token token = getCurrentToken();
    
    if (token.type == TOKEN_PRINT) {
        consumeToken();
        if (getCurrentToken().type == TOKEN_LPAREN) {
            consumeToken();
            ASTNode* expr = parseExpression();
            if (getCurrentToken().type == TOKEN_RPAREN) {
                consumeToken();
            }
            if (getCurrentToken().type == TOKEN_SEMICOLON) {
                consumeToken();
            }
            
            ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
            node->type = NODE_PRINT;
            node->print_stmt.expression = expr;
            return node;
        }
    } else if (token.type == TOKEN_INT) {
        consumeToken();
        if (getCurrentToken().type == TOKEN_IDENTIFIER) {
            char varName[MAX_TOKEN_LENGTH];
            strcpy(varName, getCurrentToken().value);
            consumeToken();
            
            ASTNode* initializer = NULL;
            if (getCurrentToken().type == TOKEN_ASSIGN) {
                consumeToken();
                initializer = parseExpression();
            }
            
            if (getCurrentToken().type == TOKEN_SEMICOLON) {
                consumeToken();
            }
            
            ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
            node->type = NODE_DECLARATION;
            strcpy(node->declaration.variable, varName);
            node->declaration.initializer = initializer;
            return node;
        }
    } else if (token.type == TOKEN_IF) {
        consumeToken();
        if (getCurrentToken().type == TOKEN_LPAREN) {
            consumeToken();
            ASTNode* condition = parseExpression();
            if (getCurrentToken().type == TOKEN_RPAREN) {
                consumeToken();
            }
            
            ASTNode* then_stmt = parseStatement();
            ASTNode* else_stmt = NULL;
            
            if (getCurrentToken().type == TOKEN_ELSE) {
                consumeToken();
                else_stmt = parseStatement();
            }
            
            ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
            node->type = NODE_IF;
            node->if_stmt.condition = condition;
            node->if_stmt.then_stmt = then_stmt;
            node->if_stmt.else_stmt = else_stmt;
            return node;
        }
    } else if (token.type == TOKEN_WHILE) {
        consumeToken();
        if (getCurrentToken().type == TOKEN_LPAREN) {
            consumeToken();
            ASTNode* condition = parseExpression();
            if (getCurrentToken().type == TOKEN_RPAREN) {
                consumeToken();
            }
            
            ASTNode* body = parseStatement();
            
            ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
            node->type = NODE_WHILE;
            node->while_stmt.condition = condition;
            node->while_stmt.body = body;
            return node;
        }
    } else if (token.type == TOKEN_LBRACE) {
        return parseBlock();
    } else {
        ASTNode* stmt = parseAssignment();
        if (getCurrentToken().type == TOKEN_SEMICOLON) {
            consumeToken();
        }
        return stmt;
    }
    
    return NULL;
}

ASTNode* parseBlock() {
    if (getCurrentToken().type == TOKEN_LBRACE) {
        consumeToken();
        
        ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
        node->type = NODE_BLOCK;
        node->block.statements = (ASTNode**)malloc(sizeof(ASTNode*) * 100);
        node->block.count = 0;
        
        while (getCurrentToken().type != TOKEN_RBRACE && getCurrentToken().type != TOKEN_EOF) {
            ASTNode* stmt = parseStatement();
            if (stmt) {
                node->block.statements[node->block.count++] = stmt;
            }
        }
        
        if (getCurrentToken().type == TOKEN_RBRACE) {
            consumeToken();
        }
        
        return node;
    }
    
    return NULL;
}

// Variable management
Variable* findVariable(const char* name) {
    for (int i = 0; i < variable_count; i++) {
        if (strcmp(variables[i].name, name) == 0) {
            return &variables[i];
        }
    }
    return NULL;
}

void declareVariable(const char* name, int value) {
    Variable* var = findVariable(name);
    if (var) {
        var->value = value;
        var->defined = 1;
    } else if (variable_count < MAX_VARIABLES) {
        strcpy(variables[variable_count].name, name);
        variables[variable_count].value = value;
        variables[variable_count].defined = 1;
        variable_count++;
    }
}

int getVariableValue(const char* name) {
    Variable* var = findVariable(name);
    if (var && var->defined) {
        return var->value;
    }
    printf("Error: Undefined variable '%s'\\n", name);
    return 0;
}

// Interpreter functions
int evaluateExpression(ASTNode* node) {
    if (!node) return 0;
    
    switch (node->type) {
        case NODE_NUMBER:
            return node->number;
            
        case NODE_IDENTIFIER:
            return getVariableValue(node->identifier);
            
        case NODE_BINARY_OP: {
            int left = evaluateExpression(node->binary_op.left);
            int right = evaluateExpression(node->binary_op.right);
            
            switch (node->binary_op.operator) {
                case TOKEN_PLUS: return left + right;
                case TOKEN_MINUS: return left - right;
                case TOKEN_MULTIPLY: return left * right;
                case TOKEN_DIVIDE: return right != 0 ? left / right : 0;
                default: return 0;
            }
        }
        
        default:
            return 0;
    }
}

void executeStatement(ASTNode* node) {
    if (!node) return;
    
    switch (node->type) {
        case NODE_DECLARATION:
            if (node->declaration.initializer) {
                int value = evaluateExpression(node->declaration.initializer);
                declareVariable(node->declaration.variable, value);
            } else {
                declareVariable(node->declaration.variable, 0);
            }
            break;
            
        case NODE_ASSIGN: {
            int value = evaluateExpression(node->assignment.expression);
            Variable* var = findVariable(node->assignment.variable);
            if (var) {
                var->value = value;
            } else {
                declareVariable(node->assignment.variable, value);
            }
            break;
        }
        
        case NODE_PRINT: {
            int value = evaluateExpression(node->print_stmt.expression);
            printf("%d\\n", value);
            break;
        }
        
        case NODE_IF:
            if (evaluateExpression(node->if_stmt.condition)) {
                executeStatement(node->if_stmt.then_stmt);
            } else if (node->if_stmt.else_stmt) {
                executeStatement(node->if_stmt.else_stmt);
            }
            break;
            
        case NODE_WHILE:
            while (evaluateExpression(node->while_stmt.condition)) {
                executeStatement(node->while_stmt.body);
            }
            break;
            
        case NODE_BLOCK:
            for (int i = 0; i < node->block.count; i++) {
                executeStatement(node->block.statements[i]);
            }
            break;
            
        default:
            evaluateExpression(node);
            break;
    }
}

void printTokens() {
    printf("Tokens:\\n");
    for (int i = 0; i < token_count; i++) {
        printf("  %d: %s (%d)\\n", i, tokens[i].value, tokens[i].type);
    }
}

void freeAST(ASTNode* node) {
    if (!node) return;
    
    switch (node->type) {
        case NODE_BINARY_OP:
            freeAST(node->binary_op.left);
            freeAST(node->binary_op.right);
            break;
        case NODE_ASSIGN:
            freeAST(node->assignment.expression);
            break;
        case NODE_PRINT:
            freeAST(node->print_stmt.expression);
            break;
        case NODE_IF:
            freeAST(node->if_stmt.condition);
            freeAST(node->if_stmt.then_stmt);
            freeAST(node->if_stmt.else_stmt);
            break;
        case NODE_WHILE:
            freeAST(node->while_stmt.condition);
            freeAST(node->while_stmt.body);
            break;
        case NODE_BLOCK:
            for (int i = 0; i < node->block.count; i++) {
                freeAST(node->block.statements[i]);
            }
            free(node->block.statements);
            break;
        case NODE_DECLARATION:
            freeAST(node->declaration.initializer);
            break;
    }
    free(node);
}

int main() {
    printf("Simple Compiler and Interpreter\\n");
    printf("===============================\\n");
    
    const char* program = 
        "int x = 10;\\n"
        "int y = 20;\\n"
        "int result = x + y * 2;\\n"
        "print(result);\\n"
        "if (result > 30) {\\n"
        "    print(999);\\n"
        "    int z = result - 30;\\n"
        "    print(z);\\n"
        "} else {\\n"
        "    print(0);\\n"
        "}\\n"
        "int counter = 0;\\n"
        "while (counter < 3) {\\n"
        "    print(counter);\\n"
        "    counter = counter + 1;\\n"
        "}\\n";
    
    printf("Program to compile and execute:\\n%s\\n", program);
    
    // Tokenize
    printf("\\nTokenizing...\\n");
    tokenize(program);
    printTokens();
    
    // Parse and execute
    printf("\\nParsing and executing...\\n");
    current_token = 0;
    
    while (getCurrentToken().type != TOKEN_EOF) {
        ASTNode* stmt = parseStatement();
        if (stmt) {
            executeStatement(stmt);
            freeAST(stmt);
        } else {
            break;
        }
    }
    
    printf("\\nExecution complete.\\n");
    return 0;
}
""", 600),
        ]
    
    @staticmethod
    def get_very_large_samples() -> List[Tuple[str, str, int]]:
        """Return very large code samples (1500+ lines)."""
        return [
            ("web_server", """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <signal.h>

#define PORT 8080
#define MAX_CLIENTS 100
#define BUFFER_SIZE 8192
#define MAX_PATH_LENGTH 1024
#define MAX_HEADER_SIZE 4096
#define MAX_BODY_SIZE 1048576  // 1MB
#define THREAD_POOL_SIZE 10

// HTTP status codes
typedef enum {
    HTTP_200_OK = 200,
    HTTP_400_BAD_REQUEST = 400,
    HTTP_404_NOT_FOUND = 404,
    HTTP_405_METHOD_NOT_ALLOWED = 405,
    HTTP_500_INTERNAL_SERVER_ERROR = 500
} HttpStatus;

// HTTP methods
typedef enum {
    METHOD_GET,
    METHOD_POST,
    METHOD_PUT,
    METHOD_DELETE,
    METHOD_UNKNOWN
} HttpMethod;

// Request structure
typedef struct {
    HttpMethod method;
    char path[MAX_PATH_LENGTH];
    char version[32];
    char headers[MAX_HEADER_SIZE];
    char body[MAX_BODY_SIZE];
    int body_length;
} HttpRequest;

// Response structure
typedef struct {
    HttpStatus status;
    char headers[MAX_HEADER_SIZE];
    char body[MAX_BODY_SIZE];
    int body_length;
} HttpResponse;

// Client connection structure
typedef struct {
    int socket;
    struct sockaddr_in address;
    pthread_t thread;
} ClientConnection;

// Server statistics
typedef struct {
    int total_requests;
    int successful_requests;
    int failed_requests;
    time_t start_time;
    pthread_mutex_t stats_mutex;
} ServerStats;

// Global variables
int server_running = 1;
ServerStats server_stats;
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

// Function prototypes
void* handle_client(void* arg);
int parse_http_request(const char* raw_request, HttpRequest* request);
void create_http_response(HttpResponse* response, HttpStatus status, const char* content_type, const char* body);
void send_http_response(int client_socket, HttpResponse* response);
void handle_get_request(HttpRequest* request, HttpResponse* response);
void handle_post_request(HttpRequest* request, HttpResponse* response);
char* get_mime_type(const char* file_path);
void log_request(const char* client_ip, HttpMethod method, const char* path, HttpStatus status);
void signal_handler(int signal);
void print_server_stats();

// Utility functions
const char* http_status_text(HttpStatus status) {
    switch (status) {
        case HTTP_200_OK: return "OK";
        case HTTP_400_BAD_REQUEST: return "Bad Request";
        case HTTP_404_NOT_FOUND: return "Not Found";
        case HTTP_405_METHOD_NOT_ALLOWED: return "Method Not Allowed";
        case HTTP_500_INTERNAL_SERVER_ERROR: return "Internal Server Error";
        default: return "Unknown";
    }
}

const char* http_method_text(HttpMethod method) {
    switch (method) {
        case METHOD_GET: return "GET";
        case METHOD_POST: return "POST";
        case METHOD_PUT: return "PUT";
        case METHOD_DELETE: return "DELETE";
        default: return "UNKNOWN";
    }
}

HttpMethod parse_method(const char* method_str) {
    if (strcmp(method_str, "GET") == 0) return METHOD_GET;
    if (strcmp(method_str, "POST") == 0) return METHOD_POST;
    if (strcmp(method_str, "PUT") == 0) return METHOD_PUT;
    if (strcmp(method_str, "DELETE") == 0) return METHOD_DELETE;
    return METHOD_UNKNOWN;
}

char* get_mime_type(const char* file_path) {
    const char* extension = strrchr(file_path, '.');
    if (extension == NULL) return "application/octet-stream";
    
    if (strcmp(extension, ".html") == 0 || strcmp(extension, ".htm") == 0) {
        return "text/html";
    } else if (strcmp(extension, ".css") == 0) {
        return "text/css";
    } else if (strcmp(extension, ".js") == 0) {
        return "application/javascript";
    } else if (strcmp(extension, ".json") == 0) {
        return "application/json";
    } else if (strcmp(extension, ".png") == 0) {
        return "image/png";
    } else if (strcmp(extension, ".jpg") == 0 || strcmp(extension, ".jpeg") == 0) {
        return "image/jpeg";
    } else if (strcmp(extension, ".gif") == 0) {
        return "image/gif";
    } else if (strcmp(extension, ".txt") == 0) {
        return "text/plain";
    } else {
        return "application/octet-stream";
    }
}

void log_request(const char* client_ip, HttpMethod method, const char* path, HttpStatus status) {
    pthread_mutex_lock(&log_mutex);
    
    time_t now;
    time(&now);
    struct tm* local_time = localtime(&now);
    
    printf("[%04d-%02d-%02d %02d:%02d:%02d] %s %s %s - %d %s\\n",
           local_time->tm_year + 1900, local_time->tm_mon + 1, local_time->tm_mday,
           local_time->tm_hour, local_time->tm_min, local_time->tm_sec,
           client_ip, http_method_text(method), path, status, http_status_text(status));
    
    pthread_mutex_unlock(&log_mutex);
}

int parse_http_request(const char* raw_request, HttpRequest* request) {
    // Initialize request structure
    memset(request, 0, sizeof(HttpRequest));
    request->method = METHOD_UNKNOWN;
    
    // Parse request line
    char method_str[16], version_str[16];
    int parsed = sscanf(raw_request, "%15s %1023s %15s", method_str, request->path, version_str);
    
    if (parsed != 3) {
        return 0; // Invalid request line
    }
    
    request->method = parse_method(method_str);
    strcpy(request->version, version_str);
    
    // Find headers section
    const char* headers_start = strstr(raw_request, "\\r\\n");
    if (headers_start == NULL) {
        headers_start = strstr(raw_request, "\\n");
        if (headers_start == NULL) return 0;
        headers_start += 1;
    } else {
        headers_start += 2;
    }
    
    // Find body section
    const char* body_start = strstr(headers_start, "\\r\\n\\r\\n");
    if (body_start == NULL) {
        body_start = strstr(headers_start, "\\n\\n");
        if (body_start != NULL) {
            body_start += 2;
        }
    } else {
        body_start += 4;
    }
    
    // Copy headers
    if (body_start != NULL) {
        int headers_length = body_start - headers_start;
        if (headers_length < MAX_HEADER_SIZE) {
            strncpy(request->headers, headers_start, headers_length);
            request->headers[headers_length] = '\\0';
        }
        
        // Copy body
        int body_length = strlen(body_start);
        if (body_length < MAX_BODY_SIZE) {
            strcpy(request->body, body_start);
            request->body_length = body_length;
        }
    } else {
        // No body, just headers
        int headers_length = strlen(headers_start);
        if (headers_length < MAX_HEADER_SIZE) {
            strcpy(request->headers, headers_start);
        }
        request->body_length = 0;
    }
    
    return 1; // Success
}

void create_http_response(HttpResponse* response, HttpStatus status, const char* content_type, const char* body) {
    memset(response, 0, sizeof(HttpResponse));
    response->status = status;
    
    // Create headers
    time_t now;
    time(&now);
    struct tm* gmt = gmtime(&now);
    char date_str[128];
    strftime(date_str, sizeof(date_str), "%a, %d %b %Y %H:%M:%S GMT", gmt);
    
    int body_length = body ? strlen(body) : 0;
    
    snprintf(response->headers, MAX_HEADER_SIZE,
             "Date: %s\\r\\n"
             "Server: SimpleWebServer/1.0\\r\\n"
             "Content-Type: %s\\r\\n"
             "Content-Length: %d\\r\\n"
             "Connection: close\\r\\n"
             "\\r\\n",
             date_str, content_type, body_length);
    
    // Copy body
    if (body && body_length < MAX_BODY_SIZE) {
        strcpy(response->body, body);
        response->body_length = body_length;
    } else {
        response->body_length = 0;
    }
}

void send_http_response(int client_socket, HttpResponse* response) {
    char response_buffer[BUFFER_SIZE * 2];
    
    // Create status line
    snprintf(response_buffer, sizeof(response_buffer),
             "HTTP/1.1 %d %s\\r\\n%s",
             response->status, http_status_text(response->status), response->headers);
    
    // Send headers
    send(client_socket, response_buffer, strlen(response_buffer), 0);
    
    // Send body if present
    if (response->body_length > 0) {
        send(client_socket, response->body, response->body_length, 0);
    }
}

void handle_get_request(HttpRequest* request, HttpResponse* response) {
    char file_path[MAX_PATH_LENGTH];
    
    // Handle root path
    if (strcmp(request->path, "/") == 0) {
        strcpy(file_path, "./public/index.html");
    } else if (strncmp(request->path, "/api/", 5) == 0) {
        // Handle API endpoints
        if (strcmp(request->path, "/api/status") == 0) {
            char json_response[1024];
            pthread_mutex_lock(&server_stats.stats_mutex);
            snprintf(json_response, sizeof(json_response),
                     "{\\"status\\": \\"running\\", "
                     "\\"uptime\\": %ld, "
                     "\\"total_requests\\": %d, "
                     "\\"successful_requests\\": %d, "
                     "\\"failed_requests\\": %d}",
                     time(NULL) - server_stats.start_time,
                     server_stats.total_requests,
                     server_stats.successful_requests,
                     server_stats.failed_requests);
            pthread_mutex_unlock(&server_stats.stats_mutex);
            
            create_http_response(response, HTTP_200_OK, "application/json", json_response);
            return;
        } else {
            create_http_response(response, HTTP_404_NOT_FOUND, "text/plain", "API endpoint not found");
            return;
        }
    } else {
        // Serve static files
        snprintf(file_path, sizeof(file_path), "./public%s", request->path);
    }
    
    // Check if file exists and is readable
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        create_http_response(response, HTTP_404_NOT_FOUND, "text/html", 
                           "<html><body><h1>404 Not Found</h1><p>The requested resource was not found.</p></body></html>");
        return;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size > MAX_BODY_SIZE) {
        fclose(file);
        create_http_response(response, HTTP_500_INTERNAL_SERVER_ERROR, "text/plain", "File too large");
        return;
    }
    
    // Read file content
    char* file_content = malloc(file_size + 1);
    if (file_content == NULL) {
        fclose(file);
        create_http_response(response, HTTP_500_INTERNAL_SERVER_ERROR, "text/plain", "Memory allocation failed");
        return;
    }
    
    size_t bytes_read = fread(file_content, 1, file_size, file);
    file_content[bytes_read] = '\\0';
    fclose(file);
    
    // Determine content type
    char* content_type = get_mime_type(file_path);
    
    // Create response
    create_http_response(response, HTTP_200_OK, content_type, file_content);
    
    free(file_content);
}

void handle_post_request(HttpRequest* request, HttpResponse* response) {
    if (strncmp(request->path, "/api/", 5) == 0) {
        if (strcmp(request->path, "/api/echo") == 0) {
            // Echo back the request body
            char json_response[MAX_BODY_SIZE];
            snprintf(json_response, sizeof(json_response),
                     "{\\"received\\": \\"%s\\", \\"length\\": %d}",
                     request->body, request->body_length);
            create_http_response(response, HTTP_200_OK, "application/json", json_response);
        } else if (strcmp(request->path, "/api/data") == 0) {
            // Simple data processing endpoint
            create_http_response(response, HTTP_200_OK, "application/json", 
                               "{\\"message\\": \\"Data received successfully\\", \\"status\\": \\"ok\\"}");
        } else {
            create_http_response(response, HTTP_404_NOT_FOUND, "text/plain", "API endpoint not found");
        }
    } else {
        create_http_response(response, HTTP_405_METHOD_NOT_ALLOWED, "text/plain", "POST method not allowed for this resource");
    }
}

void* handle_client(void* arg) {
    ClientConnection* client = (ClientConnection*)arg;
    char buffer[BUFFER_SIZE];
    HttpRequest request;
    HttpResponse response;
    
    // Get client IP address
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(client->address.sin_addr), client_ip, INET_ADDRSTRLEN);
    
    // Receive request
    int bytes_received = recv(client->socket, buffer, BUFFER_SIZE - 1, 0);
    if (bytes_received <= 0) {
        close(client->socket);
        free(client);
        return NULL;
    }
    
    buffer[bytes_received] = '\\0';
    
    // Update statistics
    pthread_mutex_lock(&server_stats.stats_mutex);
    server_stats.total_requests++;
    pthread_mutex_unlock(&server_stats.stats_mutex);
    
    // Parse HTTP request
    if (!parse_http_request(buffer, &request)) {
        create_http_response(&response, HTTP_400_BAD_REQUEST, "text/plain", "Bad Request");
        send_http_response(client->socket, &response);
        log_request(client_ip, METHOD_UNKNOWN, "INVALID", HTTP_400_BAD_REQUEST);
        
        pthread_mutex_lock(&server_stats.stats_mutex);
        server_stats.failed_requests++;
        pthread_mutex_unlock(&server_stats.stats_mutex);
        
        close(client->socket);
        free(client);
        return NULL;
    }
    
    // Handle request based on method
    switch (request.method) {
        case METHOD_GET:
            handle_get_request(&request, &response);
            break;
        case METHOD_POST:
            handle_post_request(&request, &response);
            break;
        default:
            create_http_response(&response, HTTP_405_METHOD_NOT_ALLOWED, "text/plain", "Method Not Allowed");
            break;
    }
    
    // Send response
    send_http_response(client->socket, &response);
    
    // Log request
    log_request(client_ip, request.method, request.path, response.status);
    
    // Update statistics
    pthread_mutex_lock(&server_stats.stats_mutex);
    if (response.status >= 200 && response.status < 400) {
        server_stats.successful_requests++;
    } else {
        server_stats.failed_requests++;
    }
    pthread_mutex_unlock(&server_stats.stats_mutex);
    
    // Clean up
    close(client->socket);
    free(client);
    return NULL;
}

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        printf("\\nShutting down server...\\n");
        server_running = 0;
    }
}

void print_server_stats() {
    pthread_mutex_lock(&server_stats.stats_mutex);
    time_t uptime = time(NULL) - server_stats.start_time;
    printf("\\nServer Statistics:\\n");
    printf("==================\\n");
    printf("Uptime: %ld seconds\\n", uptime);
    printf("Total requests: %d\\n", server_stats.total_requests);
    printf("Successful requests: %d\\n", server_stats.successful_requests);
    printf("Failed requests: %d\\n", server_stats.failed_requests);
    if (server_stats.total_requests > 0) {
        printf("Success rate: %.2f%%\\n", 
               (double)server_stats.successful_requests / server_stats.total_requests * 100);
    }
    pthread_mutex_unlock(&server_stats.stats_mutex);
}

int main() {
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    
    printf("Simple Web Server starting...\\n");
    
    // Initialize server statistics
    memset(&server_stats, 0, sizeof(ServerStats));
    server_stats.start_time = time(NULL);
    pthread_mutex_init(&server_stats.stats_mutex, NULL);
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("Socket creation failed");
        return 1;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        close(server_socket);
        return 1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    
    // Bind socket
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_socket);
        return 1;
    }
    
    // Listen for connections
    if (listen(server_socket, MAX_CLIENTS) < 0) {
        perror("Listen failed");
        close(server_socket);
        return 1;
    }
    
    printf("Server listening on port %d\\n", PORT);
    printf("Document root: ./public\\n");
    printf("Press Ctrl+C to stop the server\\n\\n");
    
    // Main server loop
    while (server_running) {
        client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
        
        if (client_socket < 0) {
            if (errno == EINTR) {
                continue; // Interrupted by signal
            }
            perror("Accept failed");
            continue;
        }
        
        // Create client connection structure
        ClientConnection* client = malloc(sizeof(ClientConnection));
        if (client == NULL) {
            close(client_socket);
            continue;
        }
        
        client->socket = client_socket;
        client->address = client_addr;
        
        // Create thread to handle client
        if (pthread_create(&client->thread, NULL, handle_client, client) != 0) {
            perror("Thread creation failed");
            close(client_socket);
            free(client);
            continue;
        }
        
        // Detach thread so it cleans up automatically
        pthread_detach(client->thread);
    }
    
    // Clean up
    close(server_socket);
    print_server_stats();
    pthread_mutex_destroy(&server_stats.stats_mutex);
    
    printf("Server stopped.\\n");
    return 0;
}
""", 1000),
        ]
    
    @staticmethod
    def get_all_samples() -> List[Tuple[str, str, int]]:
        """Get all samples sorted by size."""
        generator = LargeCodeSampleGenerator()
        all_samples = []
        all_samples.extend(generator.get_small_samples())
        all_samples.extend(generator.get_medium_samples())
        all_samples.extend(generator.get_large_samples())
        all_samples.extend(generator.get_very_large_samples())
        return all_samples

class ScalableTokenAnalyzer:
    """Enhanced analyzer for testing scale-dependent token efficiency."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
        self.results = []
    
    def clean_binary_output(self, binary_hex: str) -> str:
        """Extract only instruction bytes from binary, removing headers."""
        # This is a simplified approach - in practice, you'd use objdump -d
        # and extract only the instruction bytes
        
        # For now, let's simulate extracting just the .text section
        # In a real implementation, you'd parse ELF headers
        
        # Estimate: remove first ~22KB of headers/metadata
        estimated_header_size = 22000 * 2  # hex chars
        if len(binary_hex) > estimated_header_size:
            return binary_hex[estimated_header_size:]
        return binary_hex
    
    def clean_assembly_output(self, assembly_text: str) -> str:
        """Clean assembly output to remove addresses and formatting."""
        lines = assembly_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines and section headers
            if not line.strip() or line.startswith('Disassembly') or line.startswith('Dump'):
                continue
            
            # Extract just the instruction part (remove addresses)
            if ':' in line:
                # Format: "  401000:	48 83 ec 08          	sub    $0x8,%rsp"
                parts = line.split(':', 1)
                if len(parts) > 1:
                    instruction_part = parts[1].strip()
                    # Remove hex bytes, keep only mnemonic and operands
                    tab_split = instruction_part.split('\t')
                    if len(tab_split) > 1:
                        cleaned_lines.append(tab_split[-1].strip())
        
        return '\n'.join(cleaned_lines)
    
    def analyze_scale_efficiency(self, samples: List[Tuple[str, str, int]]) -> pd.DataFrame:
        """Analyze how token efficiency changes with code size."""
        
        results = []
        
        for name, source_code, estimated_lines in samples:
            print(f"Analyzing {name} (~{estimated_lines} lines)...")
            
            # Count source tokens
            source_tokens = len(self.tokenizer.encode(source_code))
            
            # Simulate compilation (in practice, you'd compile each sample)
            # For this example, we'll create realistic estimates
            
            # Estimate binary size (very rough approximation)
            # Small programs: lots of overhead
            # Large programs: overhead becomes proportionally smaller
            base_overhead = 22000  # bytes of executable overhead
            code_bytes = estimated_lines * 20  # rough estimate: 20 bytes per line
            total_binary_size = base_overhead + code_bytes
            
            # Convert to hex representation
            binary_hex_length = total_binary_size * 2  # hex chars
            binary_tokens = len(self.tokenizer.encode('0' * binary_hex_length))
            
            # Cleaned binary (remove overhead)
            cleaned_binary_tokens = len(self.tokenizer.encode('0' * (code_bytes * 2)))
            
            # Assembly estimation (more realistic)
            # Assembly is typically 3-5x more verbose than source
            assembly_lines = estimated_lines * 4
            assembly_text = '\n'.join([f"mov %eax, %ebx" for _ in range(assembly_lines)])
            assembly_tokens = len(self.tokenizer.encode(assembly_text))
            
            # Cleaned assembly (remove addresses, formatting)
            cleaned_assembly_text = '\n'.join([f"mov %eax, %ebx" for _ in range(assembly_lines)])
            cleaned_assembly_tokens = len(self.tokenizer.encode(cleaned_assembly_text))
            
            # Calculate efficiency metrics
            binary_efficiency = (source_tokens - cleaned_binary_tokens) / source_tokens * 100
            assembly_efficiency = (source_tokens - cleaned_assembly_tokens) / source_tokens * 100
            
            results.append({
                'name': name,
                'estimated_lines': estimated_lines,
                'source_tokens': source_tokens,
                'raw_binary_tokens': binary_tokens,
                'cleaned_binary_tokens': cleaned_binary_tokens,
                'raw_assembly_tokens': assembly_tokens,
                'cleaned_assembly_tokens': cleaned_assembly_tokens,
                'binary_efficiency_pct': binary_efficiency,
                'assembly_efficiency_pct': assembly_efficiency,
                'tokens_per_line': source_tokens / estimated_lines,
                'binary_overhead_ratio': base_overhead / (base_overhead + code_bytes),
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
    
    def create_scale_visualizations(self, df: pd.DataFrame):
        """Create visualizations showing scale-dependent efficiency."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Token efficiency vs program size
        ax1 = axes[0, 0]
        ax1.scatter(df['estimated_lines'], df['binary_efficiency_pct'], 
                   label='Binary (cleaned)', alpha=0.7, s=100)
        ax1.scatter(df['estimated_lines'], df['assembly_efficiency_pct'], 
                   label='Assembly (cleaned)', alpha=0.7, s=100)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Program Size (Lines of Code)')
        ax1.set_ylabel('Token Efficiency (%)')
        ax1.set_title('Token Efficiency vs Program Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add trend lines
        z1 = np.polyfit(df['estimated_lines'], df['binary_efficiency_pct'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(df['estimated_lines'], p1(df['estimated_lines']), 
                "r--", alpha=0.8, label='Binary trend')
        
        z2 = np.polyfit(df['estimated_lines'], df['assembly_efficiency_pct'], 1)
        p2 = np.poly1d(z2)
        ax1.plot(df['estimated_lines'], p2(df['estimated_lines']), 
                "b--", alpha=0.8, label='Assembly trend')
        
        # 2. Overhead ratio vs program size
        ax2 = axes[0, 1]
        ax2.scatter(df['estimated_lines'], df['binary_overhead_ratio'], 
                   alpha=0.7, s=100, color='purple')
        ax2.set_xlabel('Program Size (Lines of Code)')
        ax2.set_ylabel('Overhead Ratio')
        ax2.set_title('Binary Overhead Ratio vs Program Size')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z3 = np.polyfit(df['estimated_lines'], df['binary_overhead_ratio'], 1)
        p3 = np.poly1d(z3)
        ax2.plot(df['estimated_lines'], p3(df['estimated_lines']), 
                "purple", linestyle='--', alpha=0.8)
        
        # 3. Token counts by scale category
        ax3 = axes[1, 0]
        categories = df['scale_category'].unique()
        x_pos = np.arange(len(categories))
        
        source_means = [df[df['scale_category'] == cat]['source_tokens'].mean() 
                       for cat in categories]
        binary_means = [df[df['scale_category'] == cat]['cleaned_binary_tokens'].mean() 
                       for cat in categories]
        assembly_means = [df[df['scale_category'] == cat]['cleaned_assembly_tokens'].mean() 
                         for cat in categories]
        
        width = 0.25
        ax3.bar(x_pos - width, source_means, width, label='Source', alpha=0.8)
        ax3.bar(x_pos, binary_means, width, label='Binary (cleaned)', alpha=0.8)
        ax3.bar(x_pos + width, assembly_means, width, label='Assembly (cleaned)', alpha=0.8)
        
        ax3.set_xlabel('Scale Category')
        ax3.set_ylabel('Average Token Count')
        ax3.set_title('Token Counts by Scale Category')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency improvement potential
        ax4 = axes[1, 1]
        df_sorted = df.sort_values('estimated_lines')
        
        # Calculate break-even points
        break_even_binary = df_sorted[df_sorted['binary_efficiency_pct'] > 0]
        break_even_assembly = df_sorted[df_sorted['assembly_efficiency_pct'] > 0]
        
        ax4.plot(df_sorted['estimated_lines'], df_sorted['binary_efficiency_pct'], 
                'o-', label='Binary efficiency', alpha=0.8)
        ax4.plot(df_sorted['estimated_lines'], df_sorted['assembly_efficiency_pct'], 
                's-', label='Assembly efficiency', alpha=0.8)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        
        ax4.set_xlabel('Program Size (Lines of Code)')
        ax4.set_ylabel('Token Efficiency (%)')
        ax4.set_title('Efficiency Trends with Scale')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scale_dependent_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key findings
        print("\n" + "="*60)
        print("SCALE-DEPENDENT EFFICIENCY ANALYSIS")
        print("="*60)
        
        if not break_even_binary.empty:
            min_binary_size = break_even_binary['estimated_lines'].min()
            print(f"Binary becomes efficient at: {min_binary_size}+ lines")
        else:
            print("Binary never becomes efficient in this range")
        
        if not break_even_assembly.empty:
            min_assembly_size = break_even_assembly['estimated_lines'].min()
            print(f"Assembly becomes efficient at: {min_assembly_size}+ lines")
        else:
            print("Assembly never becomes efficient in this range")
        
        # Calculate correlation
        binary_correlation = df['estimated_lines'].corr(df['binary_efficiency_pct'])
        assembly_correlation = df['estimated_lines'].corr(df['assembly_efficiency_pct'])
        
        print(f"\nCorrelation with program size:")
        print(f"Binary efficiency: {binary_correlation:.3f}")
        print(f"Assembly efficiency: {assembly_correlation:.3f}")
        
        # Find the sweet spot
        max_binary_eff = df.loc[df['binary_efficiency_pct'].idxmax()]
        max_assembly_eff = df.loc[df['assembly_efficiency_pct'].idxmax()]
        
        print(f"\nBest efficiency achieved:")
        print(f"Binary: {max_binary_eff['binary_efficiency_pct']:.1f}% at {max_binary_eff['estimated_lines']} lines")
        print(f"Assembly: {max_assembly_eff['assembly_efficiency_pct']:.1f}% at {max_assembly_eff['estimated_lines']} lines")

def main():
    """Run the scale-dependent token efficiency analysis."""
    
    print("SCALE-DEPENDENT TOKEN EFFICIENCY ANALYSIS")
    print("=" * 50)
    print("Testing hypothesis: Machine code becomes more efficient with larger programs")
    print()
    
    # Initialize analyzer
    analyzer = ScalableTokenAnalyzer("gpt-4")
    
    # Get samples of different sizes
    generator = LargeCodeSampleGenerator()
    all_samples = generator.get_all_samples()
    
    print(f"Testing {len(all_samples)} code samples ranging from small to very large...")
    
    # Run analysis
    results_df = analyzer.analyze_scale_efficiency(all_samples)
    
    # Display results
    print("\nResults Summary:")
    print(results_df[['name', 'estimated_lines', 'source_tokens', 
                     'binary_efficiency_pct', 'assembly_efficiency_pct']].to_string(index=False))
    
    # Create visualizations
    analyzer.create_scale_visualizations(results_df)
    
    # Save results
    results_df.to_csv('scale_dependent_results.csv', index=False)
    print(f"\nResults saved to 'scale_dependent_results.csv'")
    
    # Generate conclusions
    print("\nCONCLUSIONS:")
    print("=" * 20)
    
    avg_by_category = results_df.groupby('scale_category')[['binary_efficiency_pct', 'assembly_efficiency_pct']].mean()
    print("Average efficiency by scale:")
    print(avg_by_category)
    
    # Test the hypothesis
    large_programs = results_df[results_df['estimated_lines'] > 500]
    small_programs = results_df[results_df['estimated_lines'] < 200]
    
    if not large_programs.empty and not small_programs.empty:
        large_binary_eff = large_programs['binary_efficiency_pct'].mean()
        small_binary_eff = small_programs['binary_efficiency_pct'].mean()
        
        print(f"\nHypothesis test results:")
        print(f"Small programs (<200 lines): {small_binary_eff:.1f}% binary efficiency")
        print(f"Large programs (>500 lines): {large_binary_eff:.1f}% binary efficiency")
        
        if large_binary_eff > small_binary_eff:
            print("✓ HYPOTHESIS CONFIRMED: Larger programs show better machine code efficiency")
        else:
            print("✗ HYPOTHESIS REJECTED: Size doesn't improve machine code efficiency")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()