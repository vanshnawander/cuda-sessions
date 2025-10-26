# Matrix Multiplication Basic

This directory demonstrates optimized matrix multiplication using advanced CUDA techniques like shared memory tiling.

## Files

- `TiledMatrixMul.cu`: CUDA implementation of tiled matrix multiplication for improved performance.
- `test.ipynb`: Jupyter notebook for testing the matrix multiplication kernel, including benchmarking and visualization.

## How to Execute

Compile the CUDA file:

```
nvcc TiledMatrixMul.cu -o TiledMatrixMul
./TiledMatrixMul
```

For interactive testing and analysis, open and run `test.ipynb` in Jupyter Notebook.

## What It Contains

- Use of shared memory for data reuse.
- Tiling/blocking techniques to optimize memory access patterns.
- Performance optimization in CUDA kernels.
- Benchmarking GPU vs CPU implementations.
- Debugging and profiling CUDA code.