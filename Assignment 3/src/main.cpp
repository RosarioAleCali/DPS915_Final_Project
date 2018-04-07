#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include "kernels.h"
#include <functional>
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

void reportTime(const char* msg, std::chrono::steady_clock::duration span) {
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}

int main(int argc, char* argv[]) {
	std::string file = argv[1];
	auto ts = std::chrono::steady_clock::now();
	Mat img = imread(file, -1);
	Mat temp = imread(file, -1);
	if (img.empty())
		std::cout << "failed to open img.jpg" << std::endl;
	else
		std::cout << "img.jpg loaded OK" << std::endl;
	Vec3b color;
	float blue, green, red;
	const int row_limit = img.rows;
	const int column_limit = img.cols;
	int error_code, called = 0;
	std::cout << "x: " << row_limit << std::endl;
	std::cout << "y: " << column_limit << std::endl;
	filter(img, temp, img.cols, img.rows, atoi(argv[2]));
	namedWindow("box filter", WINDOW_AUTOSIZE);
	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("box filter", temp);
	imshow("original", img);
	auto te = std::chrono::steady_clock::now();
	reportTime("Computation", te - ts);
	waitKey(0);
	destroyWindow("Lab 3");
}



