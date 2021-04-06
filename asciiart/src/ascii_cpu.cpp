#include "ascii.hpp"

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    // Image vide
    if (image.empty() || !image.data) missing_data();

    // Ouverture du stream vers le fichier de sortie
    std::ofstream output("ascii_cpu.txt");
    // Convertion de l'image, et écriture dans le fichier de sortie
    process(image, output);

    // Fermeture du stream de sortie
    output.close();

    return EXIT_SUCCESS;
}

/**
 * Fonction permettant d'obtenir le caractère ASCII correspondant à l'intensité donnée
 * @param intensity L'intensité du caractère à convertie
 * @return Le caractère ASCII correspondant
 */
char convert_intensity(uchar intensity) {
    // Convertion de l'intensité en indice dans le set de caractère
    int rounded = static_cast<int>(static_cast<float>(intensity) / divider);
    // Vérification que l'indice n'est pas OOB
    assert(rounded < chars.size());
    // Retourne le caractère correspondant
    return chars[rounded];
}

/**
 * Fonction de transformation de l'image d'entrée en ASCII
 * @param image L'image source
 * @param output Le stream vers le fichier de sortie
 */
void process(const_mat_ref image, std::ofstream& output) {

    // Parcours des ligne
    for (size_t row = 0; row < image.rows; row++) {
        // Parcours de colonnes
        for (size_t col = 0; col < image.cols; col++) {
            // Récupération du caractère correpondant au pixel actuel
            uchar converted_char = convert_intensity(image.at<uchar>(row, col));
            // Écriture dans le fichier de sortie
            output << converted_char << converted_char << converted_char;
        }
        // Ajout du fin de ligne
        output << std::endl;
    }
}