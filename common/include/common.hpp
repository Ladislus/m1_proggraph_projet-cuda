#ifndef CUDA_RIBARDIERE_WALCAK_COMMON
#define CUDA_RIBARDIERE_WALCAK_COMMON

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

typedef const cv::Mat* const_mat_ptr;
typedef const cv::Mat& const_mat_ref;

void missing_argument();
void missing_data();

#endif
