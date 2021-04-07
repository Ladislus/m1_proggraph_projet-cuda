#include "common.hpp"
#include <stdio.h>

__constant__ const size_t device_channel_number = 4;
__constant__ const size_t device_kernel_size = 9;
__constant__ const char device_coordinates[device_kernel_size][2] {
    {-1, -1}, {-1, 0}, {-1, +1},
    {0, -1}, {0, 0}, {0, +1},
    {+1, -1}, {+1, 0}, {+1, +1}
};

/**
 * Fonction permettant de vérifier que le pixel souhaité n'est pas hors de l'image
 * @param i La position en hauteur
 * @param j La position en largeur
 * @param current_coords L'indice du décalage à appliquer à la position actuel pour trouver le pixel
 * @param max_row La hauteur maximale de l'image
 * @param max_col La largeur maximale de l'image
 * @return True si le pixel est dans l'image, faux s'il est hors limite
 */
__device__
bool device_check(int i, int j, int current_coords, int max_row, int max_col) {
    int new_x = static_cast<int>(i) + device_coordinates[current_coords][0];
    int new_y = static_cast<int>(j) + device_coordinates[current_coords][1];
    // Vérification que les coordonnées sont dans les limites
    return (0 <= new_x && new_x < max_col) && (0 <= new_y && new_y < max_row);
}

/**
 * Fonction appliquant l'effet sur l'image
 * @param mat L'image source
 * @param kernel La matrice de convolution
 * @param divider Le diviseur des sommes
 * @param offset Le décalage des sommes
 * @return Un pointer vers la nouvelle image
 */
 __global__
void device_apply(const uchar* data, uchar* candidate, size_t rows, size_t cols, const int kernel[device_kernel_size], float divider, float offset) {

     uint i = blockIdx.x * blockDim.x + threadIdx.x;
     uint j = blockIdx.y * blockDim.y + threadIdx.y;

     printf("[%d; %d]\n", i, j);

     if(i < cols && j < rows) {
         // Initialisation de la somme
         int sum_blue = 0;
         int sum_green = 0;
         int sum_red = 0;

         // Pour chacun des 9 cases dans son voisinage...
         // (size_t provoque des "narrow conversion")
         for (size_t current_neighbor_index = 0; current_neighbor_index < device_kernel_size; current_neighbor_index++) {
             printf("[%d; %d] nighbor: %d\n", i, j, current_neighbor_index);

             // Si la case n'est pas hors limite...
             if (device_check(i, j, current_neighbor_index, rows, cols)) {
                 printf("[%d; %d] nighbor: %d OKOK\n", i, j, current_neighbor_index);
                 // Récupération du facteur courant (dans le kernel)
                 int current_factor = kernel[current_neighbor_index];
                 // Calcul des coordonnées du pixel à trouver
                 printf("[%d; %d] nighbor:%d OK factor:%d\n", i, j, current_neighbor_index, current_factor);

                 int new_x = static_cast<int>(i) + device_coordinates[current_neighbor_index][0];
                 int new_y = static_cast<int>(j) + device_coordinates[current_neighbor_index][1];
                 printf("[%d; %d] nighbor:%d OK factor:%d nx:%d ny:%d\n", i, j, current_neighbor_index, current_factor, new_x, new_y);

                 if (new_x >= cols || new_x < 0 || new_y < 0 || new_y >= rows) printf("[%d; %d]\n", i, j);

                 // Récupération du pixel
                 uchar blue = data[device_channel_number * (new_y * cols + new_x)];
                 uchar green = data[device_channel_number * (new_y * cols + new_x) + 1];
                 uchar red = data[device_channel_number * (new_y * cols + new_x) + 2];

                 // Ajout dans les sommes des 3 channels
                 sum_blue += blue * current_factor;
                 sum_green += green * current_factor;
                 sum_red = red * current_factor;
             }
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
         candidate[device_channel_number * (j * cols + i)] = channel_blue;
         candidate[device_channel_number * (j * cols + i) + 1] = channel_green;
         candidate[device_channel_number * (j * cols + i) + 2] = channel_red;
         candidate[device_channel_number * (j * cols + i) + 3] = data[device_channel_number * (j * cols + i) + 3];
     }
}

void flou_gaussien(const_mat_ref mat, uchar* input, uchar* output) {
    //128 threads par blocks
    dim3 thread_size( 32, 4 );
    // Calcule du nombre de block
    dim3 block_size( (( mat.cols - 1) / (thread_size.x - 2) + 1), (( mat.rows - 1 ) / (thread_size.y - 2) + 1) );

    int kernel[device_kernel_size] { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    device_apply<<<block_size, thread_size, thread_size.x * thread_size.y>>>(input, output, mat.rows, mat.cols, kernel, 16.0f, 0.0f);
}

void flou_box(const_mat_ref mat, uchar* input, uchar* output) {
    //128 threads par blocks
    dim3 thread_size( 32, 4 );
    // Calcule du nombre de block
    dim3 block_size( (( mat.cols - 1) / (thread_size.x - 2) + 1), (( mat.rows - 1 ) / (thread_size.y - 2) + 1) );

    int kernel[device_kernel_size] { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    device_apply<<<block_size, thread_size, thread_size.x * thread_size.y>>>(input, output, mat.rows, mat.cols, kernel, 9.0f, 0.0f);
}

void detection_bord(const_mat_ref mat, uchar* input, uchar* output) {
    //128 threads par blocks
    dim3 thread_size( 32, 4 );
    // Calcule du nombre de block
    dim3 block_size( (( mat.cols - 1) / (thread_size.x - 2) + 1), (( mat.rows - 1 ) / (thread_size.y - 2) + 1) );

    int kernel[device_kernel_size] {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    device_apply<<<block_size, thread_size, thread_size.x * thread_size.y>>>(input, output, mat.rows, mat.cols, kernel, 1.0f, 0.0f);
}

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    size_t data_size = image.rows * image.cols * device_channel_number;

    // Image vide
    if (image.empty() || !image.data) missing_data();
    // Image qui n'est pas sur  channels
    if (image.channels() != 4) unsupported_channel_number();

    // Pointers de l'image source sur le devide + allocation
    uchar* rgba_data = nullptr;
    cudaError e0 = cudaMalloc(&rgba_data, data_size);
    if (e0 != cudaSuccess) std::cerr << "Error 0 : " << cudaGetErrorString(e0) << std::endl;

    // Pointers de l'image de retour sur le devide + allocation
    uchar* convolution;
    cudaError e1 = cudaMalloc(&convolution, data_size);
    if (e1 != cudaSuccess) std::cerr << "Error 1 : " << cudaGetErrorString(e1) << std::endl;

    // Copie de l'image source vers le device
    cudaError e2 = cudaMemcpy(rgba_data, image.data, data_size, cudaMemcpyHostToDevice);
    if (e2 != cudaSuccess) std::cerr << "Error 2 : " << cudaGetErrorString(e2) << std::endl;

    // TIMERS avant
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lancement du calcul
    detection_bord(image, rgba_data, convolution);

    // TIMER après
    cudaEventRecord(stop);
    cudaEventSynchronize( stop );
    // Calcul du temps d'execution
    float duration;
    cudaEventElapsedTime( &duration, start, stop );
    std::cout << "Processing took: " << duration << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Pointers local de l'image de retour
    auto* output_data = new uchar[data_size];
    // Copie de l'image de retour depuis le device vers le locale
    cudaError e3 = cudaMemcpy(output_data, convolution, data_size, cudaMemcpyDeviceToHost);
    if (e3 != cudaSuccess) std::cerr << "Error 3 : " << cudaGetErrorString(e3) << std::endl;
    // Création de l'image correspondante
    auto result = cv::Mat(image.rows, image.cols, CV_8UC4, output_data);
    // Écriture dans le fichier de sortie
    cv::imwrite("convolution_gpu.png", result);

    // Free les pointers
    delete[] output_data;

    // Free des pointers CUDA
    cudaFree(rgba_data);
    cudaFree(convolution);

    return EXIT_SUCCESS;
}