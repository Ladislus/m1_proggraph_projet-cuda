#ifndef CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP
#define CUDA_RIBARDIERE_WALCAK_ASCII_CPU_HPP

#include "common.hpp"
#include <fstream>

void process(const_mat_ref, std::ofstream&);
char convert_intensity(uchar);

// Set de caractères simplifiés, qui fait perdre pas mal d'informations sur l'image
//const std::string chars = " .:-=+*#%@";
// Set de caractères plus complexe, mais qui est beaucoup plus sensible au bruit
const std::string chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

// Constante diviseur pour convertir les pixels 1 channel en caractère ASCII
const float divider = 255.0f / static_cast<float>(chars.size() - 1);

#endif
