/*
 * TILED MATRIX MULTIPLICATION - CUDA Optimization Showcase
 * ========================================================
 * 
 * This implementation demonstrates key CUDA optimization concepts:
 * 
 * 1. SHARED MEMORY (Mds, Nds):
 *    - On-chip memory ~100x faster than global memory
 *    - Reduces global memory bandwidth requirements
 *    - Limited size per block (~48KB on modern GPUs)
 * 
 * 2. MEMORY COALESCING:
 *    - Consecutive threads access consecutive memory addresses
 *    - Maximizes memory bandwidth utilization
 *    - Critical for achieving peak performance
 * 
 * 3. TILING/BLOCKING:
 *    - Divides computation into smaller tiles that fit in shared memory
 *    - Reduces global memory accesses from O(N³) to O(N³/TILE_WIDTH)
 *    - Each element loaded from global memory is reused TILE_WIDTH times
 * 
 * 4. THREAD SYNCHRONIZATION:
 *    - __syncthreads() prevents race conditions
 *    - Ensures all threads complete loading before computation
 *    - Ensures computation completes before next tile overwrites data
 * 
 * 5. COLLABORATIVE LOADING:
 *    - All threads in a block cooperate to load tiles
 *    - Achieves parallel loading of data
 * 
 * 6. BANK CONFLICT AVOIDANCE:
 *    - Access patterns avoid shared memory bank conflicts
 *    - Threads access different banks simultaneously
 * 
 * 7. REGISTER OPTIMIZATION:
 *    - Accumulation variable (Pvalue) stored in registers
 *    - Fastest memory hierarchy level
 * 
 * PERFORMANCE BENEFITS:
 * - Without tiling: Each thread accesses global memory WIDTH times
 * - With tiling: Each thread accesses global memory WIDTH/TILE_WIDTH times
 * - Speedup factor: ~TILE_WIDTH (10-20x improvement typical)
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Matrix dimensions
#define WIDTH 1024

// Kernel for tiled matrix multiplication
// C = M * N
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    // ===== CONCEPT 1: SHARED MEMORY =====
    // Shared memory is on-chip, much faster than global memory (~100x)
    // Each thread block has its own shared memory space
    // Mds and Nds store tiles of matrices M and N to reduce global memory accesses
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    // ===== CONCEPT 2: THREAD INDEXING =====
    // Block and thread indices are used to map threads to matrix elements
    int bx = blockIdx.x;  // Block index in x-dimension
    int by = blockIdx.y;  // Block index in y-dimension
    int tx = threadIdx.x; // Thread index within block (x)
    int ty = threadIdx.y; // Thread index within block (y)
    
    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty; // Global row index
    int Col = bx * TILE_WIDTH + tx; // Global column index
    
    // ===== CONCEPT 3: REGISTER USAGE =====
    // Pvalue is stored in a register (fastest memory)
    // Each thread accumulates its result in a private register
    float Pvalue = 0.0f;
    
    // ===== CONCEPT 4: TILING / BLOCKING =====
    // Loop over the M and N tiles required to compute P element
    // This reduces global memory accesses from O(Width) to O(Width/TILE_WIDTH)
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        // ===== CONCEPT 5: COLLABORATIVE LOADING =====
        // All threads in a block work together to load a tile into shared memory
        // Each thread loads one element, achieving parallel loading
        
        // ===== CONCEPT 6: MEMORY COALESCING =====
        // Loading from M: Consecutive threads (tx=0,1,2...) access consecutive memory addresses
        // This achieves coalesced memory access for optimal global memory bandwidth
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        
        // Loading from N: Consecutive threads access same column but consecutive rows
        // Column-major pattern - still coalesced because threads in same warp access
        // addresses within a cache line
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        
        // ===== CONCEPT 7: THREAD SYNCHRONIZATION =====
        // __syncthreads() is a barrier that ensures ALL threads in the block
        // have completed loading before any thread proceeds to computation
        // This prevents race conditions where some threads might try to read
        // from shared memory before others have finished writing
        __syncthreads();
        
        // ===== CONCEPT 8: SHARED MEMORY REUSE =====
        // Compute partial dot product for this tile
        // Each element from shared memory is reused TILE_WIDTH times
        // This reduces global memory accesses by a factor of TILE_WIDTH
        for (int k = 0; k < TILE_WIDTH; ++k) {
            // ===== CONCEPT 9: NO BANK CONFLICTS =====
            // Mds[ty][k]: All threads in same row access different columns (k varies)
            // Nds[k][tx]: All threads access different rows
            // Both access patterns avoid bank conflicts in shared memory
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        // ===== CONCEPT 10: BARRIER SYNCHRONIZATION =====
        // Second __syncthreads() ensures all threads finish computation
        // before the shared memory is overwritten in the next iteration
        __syncthreads();
    }
    
    // ===== CONCEPT 11: COALESCED WRITE =====
    // Write the computed value to global memory
    // Consecutive threads write to consecutive memory locations
    // This ensures coalesced write operations for optimal bandwidth
    P[Row * Width + Col] = Pvalue;
}

// ===== NAIVE GPU KERNEL (for comparison) =====
// Simple matrix multiplication without tiling
// Each thread computes one element with direct global memory access
// This demonstrates the performance difference when NOT using shared memory
__global__ void MatrixMulNaive(float* M, float* N, float* P, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (Row < Width && Col < Width) {
        float Pvalue = 0.0f;
        // Direct access to global memory - no tiling, no shared memory
        // Each element of M and N is loaded from global memory for every computation
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }
        P[Row * Width + Col] = Pvalue;
    }
}

// Host matrix multiplication for verification
void MatrixMulOnHost(float* M, float* N, float* P, int Width) {
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < Width; k++) {
                sum += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = sum;
        }
    }
}

// Initialize matrix with random values
void InitializeMatrix(float* matrix, int Width) {
    for (int i = 0; i < Width * Width; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

// Verify results
bool VerifyResult(float* hostResult, float* deviceResult, int Width) {
    float epsilon = 0.01f;
    for (int i = 0; i < Width * Width; i++) {
        if (fabs(hostResult[i] - deviceResult[i]) > epsilon) {
            printf("Mismatch at index %d: host=%f, device=%f\n", 
                   i, hostResult[i], deviceResult[i]);
            return false;
        }
    }
    return true;
}

// Function to measure execution time (cross-platform)
double get_time() {
    static const auto t0 = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = now - t0;
    return diff.count();
}

int main() {
    int Width = WIDTH;
    size_t size = Width * Width * sizeof(float);
    
    printf("========================================\n");
    printf("Matrix Multiplication Performance Test\n");
    printf("========================================\n");
    printf("Matrix Size: %dx%d\n", Width, Width);
    printf("Tile Width: %d\n", TILE_WIDTH);
    printf("========================================\n\n");
    
    // Allocate host memory
    float* h_M = (float*)malloc(size);
    float* h_N = (float*)malloc(size);
    float* h_P_cpu = (float*)malloc(size);
    float* h_P_naive = (float*)malloc(size);
    float* h_P_tiled = (float*)malloc(size);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    InitializeMatrix(h_M, Width);
    InitializeMatrix(h_N, Width);
    
    // Allocate device memory
    float *d_M, *d_N, *d_P_naive, *d_P_tiled;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P_naive, size);
    cudaMalloc((void**)&d_P_tiled, size);
    
    // Copy data to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    
    // ===== CONCEPT 12: EXECUTION CONFIGURATION =====
    // Setup execution configuration
    // Block size: TILE_WIDTH x TILE_WIDTH threads (256 threads for 16x16)
    // Each block computes one TILE_WIDTH x TILE_WIDTH tile of output matrix
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    
    // Grid size: (Width/TILE_WIDTH) x (Width/TILE_WIDTH) blocks
    // Total threads = Grid size x Block size = Width x Width
    // This maps exactly one thread per output matrix element
    dim3 dimGrid(Width/TILE_WIDTH, Width/TILE_WIDTH);
    
    printf("Grid(%d,%d) and Block(%d,%d)\n\n",
           dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    
    // ===== WARM-UP RUNS =====
    printf("Performing warm-up runs...\n");
    // Skip CPU warm-up for large matrices (too slow)
    if (Width <= 1024) {
        MatrixMulOnHost(h_M, h_N, h_P_cpu, Width);
    }
    MatrixMulNaive<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_naive, Width);
    cudaDeviceSynchronize();
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_tiled, Width);
    cudaDeviceSynchronize();
    printf("Warm-up complete.\n\n");
    
    // ===== BENCHMARK CPU =====
    double cpu_avg_time = 0.0;
    double cpu_gflops = 0.0;
    bool run_cpu_benchmark = (Width <= 2048); // Only run CPU for smaller matrices
    
    if (run_cpu_benchmark) {
        // Adaptive number of runs based on matrix size
        int num_cpu_runs = (Width <= 512) ? 10 : (Width <= 1024) ? 5 : 1;
        printf("[1/3] Benchmarking CPU implementation (%d run%s)...\n", 
               num_cpu_runs, num_cpu_runs > 1 ? "s" : "");
        
        double cpu_total_time = 0.0;
        for (int i = 0; i < num_cpu_runs; i++) {
            printf("  Run %d/%d...\r", i+1, num_cpu_runs);
            fflush(stdout);
            double start_time = get_time();
            MatrixMulOnHost(h_M, h_N, h_P_cpu, Width);
            double end_time = get_time();
            cpu_total_time += end_time - start_time;
        }
        cpu_avg_time = cpu_total_time / num_cpu_runs;
        printf("  Completed!       \n");
    } else {
        printf("[1/3] Skipping CPU benchmark (matrix too large - would take hours)\n");
        // Run once for verification only
        printf("  Running single CPU iteration for verification...\n");
        MatrixMulOnHost(h_M, h_N, h_P_cpu, Width);
        printf("  Verification data ready.\n");
    }
    
    // ===== BENCHMARK NAIVE GPU =====
    int num_gpu_runs = 10;
    printf("[2/3] Benchmarking Naive GPU (no tiling, %d runs)...\n", num_gpu_runs);
    double naive_total_time = 0.0;
    for (int i = 0; i < num_gpu_runs; i++) {
        double start_time = get_time();
        MatrixMulNaive<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_naive, Width);
        cudaDeviceSynchronize();
        double end_time = get_time();
        naive_total_time += end_time - start_time;
    }
    double naive_avg_time = naive_total_time / num_gpu_runs;
    
    // Copy naive result for verification
    cudaMemcpy(h_P_naive, d_P_naive, size, cudaMemcpyDeviceToHost);
    
    // ===== BENCHMARK TILED GPU =====
    printf("[3/3] Benchmarking Tiled GPU (optimized, %d runs)...\n\n", num_gpu_runs);
    double tiled_total_time = 0.0;
    for (int i = 0; i < num_gpu_runs; i++) {
        double start_time = get_time();
        MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_tiled, Width);
        cudaDeviceSynchronize();
        double end_time = get_time();
        tiled_total_time += end_time - start_time;
    }
    double tiled_avg_time = tiled_total_time / num_gpu_runs;
    
    // Copy tiled result for verification
    cudaMemcpy(h_P_tiled, d_P_tiled, size, cudaMemcpyDeviceToHost);
    
    // ===== VERIFICATION =====
    printf("========================================\n");
    printf("Verification\n");
    printf("========================================\n");
    
    bool naive_correct = VerifyResult(h_P_cpu, h_P_naive, Width);
    bool tiled_correct = VerifyResult(h_P_cpu, h_P_tiled, Width);
    
    printf("Naive GPU vs CPU: %s\n", naive_correct ? "PASS ✓" : "FAIL ✗");
    printf("Tiled GPU vs CPU: %s\n\n", tiled_correct ? "PASS ✓" : "FAIL ✗");
    
    // ===== PERFORMANCE RESULTS =====
    printf("========================================\n");
    printf("Performance Results\n");
    printf("========================================\n");
    
    // Calculate GFLOPS
    double numOperations = 2.0 * Width * Width * Width;
    double naive_gflops = (numOperations / naive_avg_time) / 1e9;
    double tiled_gflops = (numOperations / tiled_avg_time) / 1e9;
    
    if (run_cpu_benchmark) {
        cpu_gflops = (numOperations / cpu_avg_time) / 1e9;
        printf("CPU Time:       %8.2f ms  (%6.2f GFLOPS)\n", cpu_avg_time * 1000, cpu_gflops);
    } else {
        printf("CPU Time:       Skipped (too slow for %dx%d)\n", Width, Width);
    }
    printf("Naive GPU Time: %8.2f ms  (%6.2f GFLOPS)\n", naive_avg_time * 1000, naive_gflops);
    printf("Tiled GPU Time: %8.2f ms  (%6.2f GFLOPS)\n\n", tiled_avg_time * 1000, tiled_gflops);
    
    printf("Speedup Analysis:\n");
    if (run_cpu_benchmark) {
        printf("  Naive GPU vs CPU:  %.2fx faster\n", cpu_avg_time / naive_avg_time);
        printf("  Tiled GPU vs CPU:  %.2fx faster\n", cpu_avg_time / tiled_avg_time);
    }
    printf("  Tiled vs Naive:    %.2fx faster\n", naive_avg_time / tiled_avg_time);
    printf("\n");
    
    printf("Optimization Benefits:\n");
    printf("  Memory access reduction: ~%dx\n", TILE_WIDTH);
    printf("  Total FLOPs computed:    %.2f billion\n", numOperations / 1e9);
    printf("========================================\n");
    
    if (!run_cpu_benchmark) {
        printf("\nNote: CPU benchmark skipped for large matrices.\n");
        printf("For %dx%d, CPU would take ~%.1f minutes per run.\n", 
               Width, Width, (naive_avg_time * 50) / 60.0); // Rough estimate
    }
    
    // Cleanup
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P_naive);
    cudaFree(d_P_tiled);
    free(h_M);
    free(h_N);
    free(h_P_cpu);
    free(h_P_naive);
    free(h_P_tiled);
    
    return 0;
}
