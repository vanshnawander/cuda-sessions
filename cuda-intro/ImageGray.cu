#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

// CUDA kernel to convert color image to grayscale
__global__ 
void colorToGrayscaleKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = col + row * width;
    
    if (col < width && row < height) {
        if (channels == 1) {
            // Already grayscale
            output[index] = input[index];
        } else if (channels == 3 || channels == 4) {
            // RGB or RGBA image
            int pixelIndex = (row * width + col) * channels;
            unsigned char r = input[pixelIndex];
            unsigned char g = input[pixelIndex + 1];
            unsigned char b = input[pixelIndex + 2];
            
            // Convert to grayscale using luminance formula
            // 0.299 * R + 0.587 * G + 0.114 * B
            output[index] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return -1;
    }
    
    const char* inputPath = argv[1];
    const char* outputPath = argv[2];
    
    // Load image using stb_image
    int width, height, channels;
    // The * indicates that inputImage is a pointer to unsigned char
    // It points to the memory location where the image data is stored
    // stbi_load returns a pointer to the loaded image data in memory
    unsigned char* inputImage = stbi_load(inputPath, &width, &height, &channels, 0);
    
    if (!inputImage) {
        std::cerr << "Error: Could not load image " << inputPath << std::endl;
        return -1;
    }
    
    cout << "Loaded image: " << width << "x" << height << " with " << channels << " channels" << endl;
    // Print input image array
    cout << "Input image data:" << endl;

    // for (int i = 0; i < width * height * channels; i++) {
    //     cout << static_cast<int>(inputImage[i]) << " ";
    //     if ((i + 1) % (width * channels) == 0) {
    //         cout << endl;
    //     }
    // }
    cout << endl;
    cout << "Input image data:" << endl;
    cout << inputImage << endl;

    
    // Allocate memory for grayscale output
    size_t inputSize = width * height * channels * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);
    
    unsigned char* d_input;
    unsigned char* d_output;
    
    // Allocate GPU memory
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
    
    // Copy input image to GPU
    cudaMemcpy(d_input, inputImage, inputSize, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    colorToGrayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        stbi_image_free(inputImage);
        return -1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Allocate host memory for output
    unsigned char* outputImage = new unsigned char[width * height];
    
    // Copy result back to host
    cudaMemcpy(outputImage, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    // Save grayscale image
    if (stbi_write_png(outputPath, width, height, 1, outputImage, width)) {
        std::cout << "Grayscale image saved to " << outputPath << std::endl;
    } else {
        std::cerr << "Error: Could not save image " << outputPath << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(inputImage);
    delete[] outputImage;
    
    return 0;
}
