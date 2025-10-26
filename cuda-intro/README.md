# CUDA Intro

This directory contains introductory CUDA examples to get started with GPU programming.

## Files

- `vectorAdd.cu`: Basic vector addition kernel demonstrating host-device memory transfer and kernel launch.
- `SimpleMatrixMultiplication.cu`: Simple matrix multiplication implementation on GPU.
- `myCuda.cu`: Basic CUDA program setup and device query.
- `ImageGray.cu`: Image grayscale conversion using CUDA kernels (requires stb_image headers).

## How to Execute

Compile each CUDA file with NVCC:

```
nvcc vectorAdd.cu -o vectorAdd
./vectorAdd
```

For image processing, ensure `stb_image.h` and `stb_image_write.h` are in the directory.

## What It Teaches

- Basic CUDA kernel syntax and launch parameters.
- Memory allocation on host and device (cudaMalloc, cudaMemcpy).
- Parallel execution concepts.
- Error handling in CUDA.
- Simple GPU-accelerated algorithms.