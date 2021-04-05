#include "convolution.hpp"

int main(int argc, char** argv) {

    if (argc != 2) missing_argument();

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    if (image.empty() || !image.data) missing_data();

    if (image.channels() != 4) throw std::logic_error("No support for image that are not 4 channels yet !");

    const_mat_ptr result = process(image, Effect::FLOU_GAUSSIEN);
    cv::imwrite("convolution_cpu.png", *result);

    delete result;

    return 0;
}

const_mat_ptr process(const_mat_ref mat, const Effect& effect) {
    try {
        return effect_functions.at(effect)(mat);
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error finding corresponding fonction" << std::endl;
        exit(EXIT_FAILURE);
    }
}

const_mat_ptr apply(const_mat_ref mat, const_vector_ref kernel, float divider, float offset) {

    auto* candidate = new cv::Mat(mat.rows, mat.cols, CV_8UC(4));

    // Pour chacunes des lignes
    // (size_t provoque des "narrow conversion")
    for (int i = 0; i < mat.rows; i++)

        // Pour chacunes de colonnes
        // (size_t provoque des "narrow conversion")
        for (int j = 0; j < mat.cols; j++) {

            // Initialisation de la somme
            int sum_blue = 0;
            int sum_green = 0;
            int sum_red = 0;

            // Pour chacun des 9 cases dans son voisinage
            // (size_t provoque des "narrow conversion")
            for (int current_neighbor_index = 0; current_neighbor_index < kernel.size(); current_neighbor_index++)

                // Si la case n'est pas hors limite
                if (check(i, j, current_neighbor_index, mat.rows, mat.cols)) {
                    int current_factor = kernel[current_neighbor_index];
                    std::pair<int, int> current_coords = {i + coordinates[current_neighbor_index].first, j + coordinates[current_neighbor_index].second };
                    cv::Vec4b current_pixel = mat.at<cv::Vec4b>(current_coords.first, current_coords.second);


                    sum_blue += current_pixel[0] * current_factor;
                    sum_green += current_pixel[1] * current_factor;
                    sum_red += current_pixel[2] * current_factor;
                }

            uchar result_blue = (sum_blue > 255) ? 255 : ((sum_blue < 0) ? 0 : (static_cast<float>(sum_blue) / divider) + offset);
            uchar result_green = (sum_green > 255) ? 255 : ((sum_green < 0) ? 0 : (static_cast<float>(sum_green) / divider) + offset);
            uchar result_red = (sum_red > 255) ? 255 : ((sum_red < 0) ? 0 : (static_cast<float>(sum_red) / divider) + offset);

            candidate->at<cv::Vec4b>(i, j) = { result_blue, result_green, result_red, mat.at<cv::Vec4b>(i, j)[3] };
        }

    return candidate;
}

bool check(int i, int j, int current_coords, int max_row, int max_col) {
    std::pair<int, int> modifier = coordinates[current_coords];
    std::pair<int, int> new_coords = { (i + modifier.first), (j + modifier.second) };
    return (0 <= new_coords.first && new_coords.first <= max_row) && (0 <= new_coords.second && new_coords.second <= max_col);
}

const_mat_ptr flou_gaussien(const_mat_ref mat) {
    return apply(mat, std::vector({ 1, 2, 1, 2, 4, 2, 1, 2, 1 }), 16.0f, 0.0f);
}

const_mat_ptr flou_box(const_mat_ref mat) {
    return apply(mat, std::vector({ 1, 1, 1, 1, 1, 1, 1, 1, 1 }), 9.0f, 0.0f);
}

const_mat_ptr detection_bord(const_mat_ref mat) {
    return apply(mat, std::vector({-1, -1, -1, -1, 8, -1, -1, -1, -1}), 1.0f, 0.0f);
}