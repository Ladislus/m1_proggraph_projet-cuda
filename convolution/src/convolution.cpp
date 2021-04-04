#include "convolution.hpp"

int main(int argc, char** argv) {

    if (argc != 2) missing_argument();

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    if (image.empty() || !image.data) missing_data();

    const_mat_ptr result = process(image, Effect::FLOU_GAUSSIEN);
    std::cout << result << std::endl;

    return 0;
}

const_mat_ptr process(const_mat_ref mat, const Effect& effect) {
    try {
        return effect_functions.at(effect)(mat);
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error finding corresponding fonction" << std::endl;
        exit(1);
    }
}

const_mat_ptr apply(const_mat_ref mat, const_vector_ref kernel, float divider, float offset) {
    // TODO
    return nullptr;
}

const_mat_ptr flou_gaussien(const_mat_ref mat) {
    return apply(mat, std::vector({ 1, 1, 1, 1, 1, 1, 1, 1, 1 }), 16.0f, 0.0f);
}

const_mat_ptr detection_bord(const_mat_ref mat) {
    return apply(mat, std::vector({-1, -1, -1, -1, 8, -1, -1, -1, -1}), 1.0f, 0.0f);
}