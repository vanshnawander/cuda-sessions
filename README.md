These are example CUDA kernels. 
you can them using NVCC compiler, make you sure you have nvcc setup properly:

nvcc filename.cu -o outputfile

for example:
nvcc SimpleMatrixMultiplication.cu -o SimpleMatrixMultiplication

then run the output file:
./SimpleMatrixMultiplication

note: make sure you have the stb_image.h and stb_image_write.h files in the same directory as the ImageGray.cu file. These are header files for image processing. I have added them in the repo itself. 