#ifndef CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP
#define CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>

void missing_argument();
void missing_data();
void process(const cv::Mat&, std::ofstream&);
char convert_intensity(uchar);

const std::string chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const float divider = 255.0f / (chars.size() - 1);

#endif
