#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
#include <chrono>
using namespace cv;
int findingNeighbors(Mat img, int i, int j, int neighbour,float * b, float * g, float * r) {
	int row_limit = img.rows;
	int column_limit = img.cols;
	Scalar temp;
	double sum = 0, blue=0, red=0, green=0;

	for (int x = i - floor(neighbour / 2); x <= i + floor(neighbour / 2); x++) {
		for (int y = j - floor(neighbour / 2); y <= j + floor(neighbour / 2); y++) {
			if (x >= 0 && y >= 0 && x < row_limit && y < column_limit) {
				temp = img.at<Vec3b>(x, y);
				blue += temp.val[0];
				green += temp.val[1];
				red += temp.val[2];
			}
		}
		
	}
	*b = blue / pow(neighbour, 2);
	*g = green / pow(neighbour, 2);
	*r = red / pow(neighbour, 2);
	return 1;
}
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
	int row_limit = img.rows;
	int column_limit = img.cols;
	int error_code,called=0;
	std::cout << "x: " << row_limit << std::endl;
	std::cout << "y: " << column_limit << std::endl;
	for (int i = 0; i < row_limit; i++) {
		for (int j = 0; j < column_limit; j++) {
			color = img.at<Vec3b>(i, j);
			error_code = findingNeighbors(img, i, j, atoi(argv[2]), &blue, &green, &red);
			temp.at<Vec3b>(i, j)[0] = blue;
			temp.at<Vec3b>(i, j)[1] = green;
			temp.at<Vec3b>(i, j)[2] = red;

			called++;
		}
	}

	imshow("box filter", temp);
	imshow("original", img);
	std::cout << "called:		" <<called << std::endl;
	auto te = std::chrono::steady_clock::now();
	reportTime("Computation", te - ts);
	waitKey(0);
	destroyWindow("Lab 3");
}