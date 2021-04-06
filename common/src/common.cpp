#include "common.hpp"

/**
 * Fonction permettant d'indiquer que des arguments sont manquants
 */
void missing_argument() {
    throw std::logic_error("Missing argument <filepath>\nusage: ./[program] <Image_Path>");
}

/**
 * Fonction permettant d'indiquer que l'image est vide
 */
void missing_data() {
    throw std::logic_error("Image couldn't be found or opened !");
}

/**
 * Fonction permettant d'indiquer que le nombre de channel n'est pas (encore) supportée
 */
void unsupported_channel_number() {
    throw std::logic_error("No support for image that are not 4 channels yet !");
}
