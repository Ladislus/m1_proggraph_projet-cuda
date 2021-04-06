#include "grayscale.hpp"

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Image vide
    if (image.empty() || !image.data) missing_data();

    // Création de la copie avec l'effet
    const_mat_ptr result = process(image);
    // Enregistrement de l'image
    cv::imwrite("grayscale_cpu.png", *result);

    // Free les pointers
    delete result;

    return EXIT_SUCCESS;
}

/**
 * Fonction pour convertir un pixel 3 channels en pixel grayscale 1 channel
 * @param pixel Le pixel à convertir
 * @return Le pixel convertie en grayscale
 */
uchar convert_intensity(cv::Vec3b pixel) {
    return (((113 * pixel[0]) + (604 * pixel[1]) + (307 * pixel[2])) / 1024);
}

/**
 * Fonction appliquant l'effet grayscale sur l'image
 * @param image L'image source
 * @return Un pointer vers la nouvelle image
 */
const_mat_ptr process(const_mat_ref image) {

    // Création de la matrice de retour (1 channel)
    auto* candidate = new cv::Mat(image.rows, image.cols, CV_8UC1);

    // Parcour des lignes
    for (size_t i = 0; i < image.rows; i++)
        // Parcour des colonnes
        for (size_t j = 0; j < image.cols; j++)
            // Convertion du pixel actuel en pixel grayscale, et ajout dans l'image de retour
            candidate->at<uchar>(i, j) = convert_intensity(image.at<cv::Vec3b>(i, j));

    return candidate;
}