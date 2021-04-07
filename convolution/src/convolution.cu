#include "convolution.hpp"

int main(int argc, char** argv) {
    return EXIT_SUCCESS;
}

const_mat_ptr process(const_mat_ref mat, const Effect& effect) {
    try {
        // Appel à la fonction correspondant à l'effet demandé
        return effect_functions.at(effect)(mat);
    } catch (const std::out_of_range& oor) {
        // Effet inconnu
        std::cerr << "Error finding corresponding fonction" << std::endl;
        exit(EXIT_FAILURE);
    }
}

const_mat_ptr apply(const_mat_ref mat, const_vector_ref kernel, float divider, float offset) {

    // Création de la matrice de retour
    return new cv::Mat(mat.rows, mat.cols, CV_8UC(4));
}

bool check(int i, int j, int current_coords, int max_row, int max_col) {
    return true;
}

const_mat_ptr flou_gaussien(const_mat_ref mat) {
    return new cv::Mat();
}

const_mat_ptr flou_box(const_mat_ref mat) {
    return new cv::Mat();
}

const_mat_ptr detection_bord(const_mat_ref mat) {
    return new cv::Mat();
}