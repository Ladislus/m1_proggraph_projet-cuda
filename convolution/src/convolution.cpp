#include "convolution.hpp"

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    // Image vide
    if (image.empty() || !image.data) missing_data();
    // Image qui n'est pas sur  channels
    if (image.channels() != 4) unsupported_channel_number();

    // Création de la copie avec l'effet
    const_mat_ptr result = process(image, Effect::DETECTION_BORD);
    // Enregistrement de l'image
    cv::imwrite("convolution_cpu.png", *result);

    // Free les pointers
    delete result;

    return EXIT_SUCCESS;
}

/**
 * Fonction permettant de créer une copie de l'image sur lequel on applique l'effet demandé.
 * @param mat La matrice de pixel sur lequel appliquer l'effet
 * @param effect Le flag correspondant à l'effet souhaité
 * @return Un pointer vers la nouvelle image
 */
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

/**
 * Fonction appliquant l'effet sur l'image
 * @param mat L'image source
 * @param kernel La matrice de convolution
 * @param divider Le diviseur des sommes
 * @param offset Le décalage des sommes
 * @return Un pointer vers la nouvelle image
 */
const_mat_ptr apply(const_mat_ref mat, const_vector_ref kernel, float divider, float offset) {

    // Création de la matrice de retour
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


            // Pour chacun des 9 cases dans son voisinage...
            // (size_t provoque des "narrow conversion")
            for (int current_neighbor_index = 0; current_neighbor_index < kernel.size(); current_neighbor_index++)

                // Si la case n'est pas hors limite...
                if (check(i, j, current_neighbor_index, mat.rows, mat.cols)) {
                    // Récupération du facteur courant (dans le kernel)
                    int current_factor = kernel[current_neighbor_index];
                    // Calcul des coordonnées du pixel à trouver
                    std::pair<int, int> current_coords = {i + coordinates[current_neighbor_index].first,
                                                          j + coordinates[current_neighbor_index].second};
                    // Récupération du pixel
                    cv::Vec4b current_pixel = mat.at<cv::Vec4b>(current_coords.first, current_coords.second);

                    // Ajout dans les sommes des 3 channels
                    sum_blue += current_pixel[0] * current_factor;
                    sum_green += current_pixel[1] * current_factor;
                    sum_red += current_pixel[2] * current_factor;
                }

            // Calcul des sommmes de convolution des 3 channels
            int result_blue = static_cast<int>((static_cast<float>(sum_blue) / divider) + offset);
            int result_green = static_cast<int>((static_cast<float>(sum_green) / divider) + offset);
            int result_red = static_cast<int>((static_cast<float>(sum_red) / divider) + offset);
            // Convertion des sommes en unsigned char (0 <= x <= 255)
            uchar channel_blue = (result_blue > 255) ? 255 : ((result_blue < 0) ? 0 : result_blue);
            uchar channel_green = (result_green > 255) ? 255 : ((result_green < 0) ? 0 : result_green);
            uchar channel_red = (result_red > 255) ? 255 : ((result_red < 0) ? 0 : result_red);
            // Mise à jour du pixel
            candidate->at<cv::Vec4b>(i, j) = {channel_blue, channel_green, channel_red, mat.at<cv::Vec4b>(i, j)[3]};
        }
    // Retourne la nouvelle matrice
    return candidate;
}

/**
 * Fonction permettant de vérifier que le pixel souhaité n'est pas hors de l'image
 * @param i La position en hauteur
 * @param j La position en largeur
 * @param current_coords L'indice du décalage à appliquer à la position actuel pour trouver le pixel
 * @param max_row La hauteur maximale de l'image
 * @param max_col La largeur maximale de l'image
 * @return True si le pixel est dans l'image, faux s'il est hors limite
 */
bool check(int i, int j, int current_coords, int max_row, int max_col) {
    // Récupéartion du décalage en (X,Y)
    std::pair<int, int> modifier = coordinates[current_coords];
    // Calcul des nouvelles coordonnées
    std::pair<int, int> new_coords = { (i + modifier.first), (j + modifier.second) };
    // Vérification que les coordonnées sont dans les limites
    return (0 <= new_coords.first && new_coords.first < max_row) && (0 <= new_coords.second && new_coords.second < max_col);
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