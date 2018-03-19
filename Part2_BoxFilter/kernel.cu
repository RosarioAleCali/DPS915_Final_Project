
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
#include <chrono>
#include <stdio.h>

using namespace cv;
const int ntpb = 32; // number of threads per block
__global__ void NeighborKernel(Mat * src, Mat * result, int i, int j, int neighbour) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int row_limit = src.rows; 
	int column_limit = src.cols;
	Scalar temp;
	double sum = 0, blue = 0, red = 0, green = 0;

	for (int i = x - floor(neighbour / 2); i <= x + floor(neighbour / 2); i++) {
		for (int j = x - floor(neighbour / 2); j <= y + floor(neighbour / 2); j++) {
			if (i >= 0 && j >= 0 && i < row_limit && j < column_limit) {
				temp = src.at<Vec3b>(x, y);
				blue += temp.val[0];
				green += temp.val[1];
				red += temp.val[2];
			}
		}
	}
	result.at<Vec3b>(x, y)[0] = blue / pow(neighbour, 2);
	result.at<Vec3b>(x, y)[1] = green / pow(neighbour, 2);
	result.at<Vec3b>(x, y)[2] = red / pow(neighbour, 2);
}
void findingNeighbors(Mat * src, Mat * result, int i, int j, int neighbour) {
	Mat *Md, *Pd;
	int nb = (i + ntpb - 1) / ntpb;
	cudaMalloc((void**)&Md, i * j * sizeof(int));
	cudaMemcpy(Md, src, i * j * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd, i * j * sizeof(int));
	dim3 dGrid(nb, nb, 1);
	dim3 dBlock(ntpb, ntpb, 1);
	NeighborKernel<<<dGrid, dBlock >>>(src, result, i, j, neighbour); 
	cudaMemcpy(result, Pd, i * j * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Pd);
}
void reportTime(const char* msg, std::chrono::steady_clock::duration span) {
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}
int main(int argc, char* argv[]) {
	std::string file = argv[1];
	auto ts = std::chrono::steady_clock::now();
	Mat src = imread(file, -1);
	Mat result = imread(file, -1);
	if (src.empty())
		std::cout << "failed to open img.jpg" << std::endl;
	else
		std::cout << "img.jpg loaded OK" << std::endl;
	Vec3b color;
	float blue, green, red;
	int row_limit = src.rows;
	int column_limit = src.cols;
	int called = 0;
	std::cout << "x: " << row_limit << std::endl;
	std::cout << "y: " << column_limit << std::endl;
	findingNeighbors(&src, &result, row_limit, column_limit, atoi(argv[2]));
	imshow("box filter", result);
	imshow("original", src);
	std::cout << "called:		" << called << std::endl;
	auto te = std::chrono::steady_clock::now();
	reportTime("Computation", te - ts);
	waitKey(0);
	destroyWindow("Lab 3");
}