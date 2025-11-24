# Inference Engine
A high-performance inference system with flexible execution strategies for computer vision models.

## Features
- **Multi-Context Inference**: Parallel execution across multiple execution contexts using different threads
- **Dynamic Threading**: Variable number of "CUDA" threads within single context threads
- **Hybrid Execution**: Combined multi-context and dynamic threading approaches
- **Reducing startup overhead**: The ability to pin the computation graph to minimize the central process's contribution to running cores.
- **Expandable Worker Pool**: Scalable worker management system
- **Synchronized Data Retrieval**: Guaranteed data consistency from running workers

## Installation
Pre-build of the Yolo engine is required after the next [repository](https://github.com/SugarBlend/-DeployAndServe/tree/main/deploy2serve/deployment/projects/yolo).
```cmd
setup.bat
```
