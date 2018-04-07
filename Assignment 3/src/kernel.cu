#include "kernels.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>


__global__
void blur(unsigned char* input_image, unsigned char* output_image, const int width, const int height, int neighbour) {
	const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int x = offset % width;
	int y = (offset - x) / width;
	if (offset < width*height) {
		float output_red = 0;
		float output_green = 0;
		float output_blue = 0;
		int hits = 0;
		for (int ox = -neighbour; ox < neighbour + 1; ++ox) {
			for (int oy = -neighbour; oy < neighbour + 1; ++oy) {
				if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height) {
					const int currentoffset = (offset + ox + oy*width) * 3;
					output_red += input_image[currentoffset];
					output_green += input_image[currentoffset + 1];
					output_blue += input_image[currentoffset + 2];
					hits++;
				}
			}
		}
		output_image[offset * 3] = static_cast<unsigned char>(output_red / hits);
		output_image[offset * 3 + 1] = static_cast<unsigned char>(output_green / hits);
		output_image[offset * 3 + 2] = static_cast<unsigned char>(output_blue / hits);
	}
}

void filter(const Mat& input, Mat& output, int width, int height, int neighbour)
{
	int pixels = height * width;
	int iDevice;
	cudaDeviceProp prop;
	cudaGetDevice(&iDevice);
	cudaGetDeviceProperties(&prop, iDevice);
	int resident_threads = prop.maxThreadsPerMultiProcessor;
	int resident_blocks = 8;
	if (prop.major >= 3 && prop.major < 5) {
		resident_blocks = 16;
	}
	else if (prop.major >= 5 && prop.major <= 6) {
		resident_blocks = 32;
	}

	//Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	cudaMalloc((void**)&d_input, width*height * 3 * sizeof(unsigned char));
	cudaMalloc((void**)&d_output, width*height * 3 * sizeof(unsigned char));

	//Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), width*height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	dim3 blockDims(resident_threads/resident_blocks,1,1);
	
	//Calculate grid size to cover the whole image
	dim3 gridDims(pixels/blockDims.x);
	//Launch the color conversion kernel
	blur << <gridDims, blockDims >> >(d_input, d_output, input.cols, input.rows, neighbour);

	//Synchronize to check for any kernel launch errors
	cudaDeviceSynchronize();

	//Copy back data from destination device meory to OpenCV output image

	cudaMemcpy(output.ptr(), d_output, width*height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);
}
