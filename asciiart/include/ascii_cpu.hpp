#ifndef CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP
#define CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP

#include <opencv2/opencv.hpp>
#include <cstdlib>

void missing_argument();
void missing_data();
cv::Mat* process(const cv::Mat&);
char convert_intensity(uchar);

const std::string chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

#endif
