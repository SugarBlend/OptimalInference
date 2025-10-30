# Inference Engine
A high-performance inference system with flexible execution strategies for computer vision models.

## Features
- **Multi-Context Inference**: Parallel execution across multiple execution contexts using different threads
- **Dynamic Threading**: Variable number of "Where" threads within single context threads  
- **Hybrid Execution**: Combined multi-context and dynamic threading approaches
- **Expandable Worker Pool**: Scalable worker management system
- **Synchronized Data Retrieval**: Guaranteed data consistency from running workers

## Performance
Example of detector inference on a batch of pre-prepared images:
<img src="assets/trace.gif" alt="Inference Performance Visualization" width="1280" height="720">

## Installation
```cmd
setup.bat
```
