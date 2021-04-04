#include "common.hpp"

void missing_argument() {
    std::cerr << "usage: ./[program] <Image_Path>" << std::endl;
    exit(EXIT_FAILURE);
}

void missing_data() {
    std::cerr << "Image couldn't be found or opened !" << std::endl;
    exit(EXIT_FAILURE);
}
