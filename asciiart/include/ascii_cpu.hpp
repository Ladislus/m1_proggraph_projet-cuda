#ifndef CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP
#define CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP

#include "common.hpp"
#include <fstream>

void process(const_mat_ref, std::ofstream&);
char convert_intensity(uchar);

const std::string chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const float divider = 255.0f / static_cast<float>(chars.size() - 1);

#endif
