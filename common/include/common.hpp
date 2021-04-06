#ifndef CUDA_RIBARDIERE_WALCAK_COMMON
#define CUDA_RIBARDIERE_WALCAK_COMMON

#include <opencv2/opencv.hpp>

typedef const cv::Mat* const_mat_ptr;
typedef const cv::Mat& const_mat_ref;

// Fonctions d'erreur
void missing_argument();
void missing_data();
void unsupported_channel_number();

#endif
