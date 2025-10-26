
#include <stdio.h>


__global__  // __global__ is a keyword that marks a function as a kernel that can be executed on the GPU
void vectorAddKernel(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; // blockDim is the block size and blockIdx is the block index
    // threadIdx is the thread index and blockIdx is the block index
    printf("Thread %d (threadIdx:%d, blockIdx:%d, blockDim:%d, gridDim:%d): %f + %f = %f\n", 
           i, threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, a[i], b[i], c[i]);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vecAdd(float *a, float *b, float *c, int n) {
    float *a_d, *b_d, *c_d;
    int size = n * sizeof(float);

    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&c_d, size);


    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);
    int blockSize = 512;
    int gridSize = (n + blockSize - 1) / blockSize;

    dim3 block(blockSize,1,1); // dim3 is type used to define the dimensions of the block and grid 
    dim3 grid(gridSize,1,1);  // grid is the number of blocks in the grid
    
    vectorAddKernel<<<grid, block>>>(a_d, b_d, c_d, n);
    cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


int main() {
    float *a, *b, *c;
    int n = 50;
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    c = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    vecAdd(a, b, c, n);
    for (int i = 0; i < n; i++) {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }
    free(a);
    free(b);
    free(c);
    return 0;
}
