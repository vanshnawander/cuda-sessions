# CUDA Sessions

This repository contains CUDA learning materials and example kernels for getting started with NVIDIA CUDA programming. It provides a progression from basic CUDA concepts to optimized implementations.

## Repository Structure

- `cuda-intro/`: Introductory CUDA examples covering basic kernels, memory management, and simple algorithms.
- `matmul-basic/`: matrix multiplication implementations demonstrating optimization techniques like tiling.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVCC compiler installed (part of CUDA Toolkit)
- stb_image.h and stb_image_write.h headers (included in the repo for image processing examples)

## How to Compile and Run CUDA Files

Compile a CUDA file using NVCC:

```
nvcc filename.cu -o outputfile
./outputfile
```

For example:

```
nvcc SimpleMatrixMultiplication.cu -o SimpleMatrixMultiplication
./SimpleMatrixMultiplication
```

Note: Ensure the stb_image headers are in the same directory as files like `ImageGray.cu` that use them.