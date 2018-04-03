#ifndef KERNELS_H_
#define KERNELS_H_
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

void filter(const Mat& input, Mat& output, int width, int height, int neighbour);


#endif
