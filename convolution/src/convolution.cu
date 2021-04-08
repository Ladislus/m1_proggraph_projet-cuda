#include "common.hpp"

// Constante sur le device pour connaitre le nombre de channels de l'image
__constant__ const size_t device_channel_number = 4;
// Constante sur le device pour connaitre la taille des kernels
__constant__ const size_t device_kernel_size = 9;
// Liste des calculs à effectuer pour obtenir les pixels voisins
__constant__ const char device_coordinates[device_kernel_size][2] {
    {-1, -1}, {-1, 0}, {-1, +1},
    {0, -1}, {0, 0}, {0, +1},
    {+1, -1}, {+1, 0}, {+1, +1}
};

/**
 * Fonction permettant de vérifier que le pixel souhaité n'est pas hors de l'image
 * @param  La position en largeur
 * @param j La position en hauteur
 * @param current_coords L'indice du décalage à appliquer à la position actuel pour trouver le pixel
 * @param max_row La hauteur maximale de l'image
 * @param max_col La largeur maximale de l'image
 * @return True si le pixel est dans l'image, faux s'il est hors limite
 */
__device__
bool device_check(uint i, uint j, uint current_coords, size_t max_row, size_t max_col) {
    // Cast en signed int pour vérifier que la soustraction est > 0
    int new_x = static_cast<int>(i) + device_coordinates[current_coords][0];
    int new_y = static_cast<int>(j) + device_coordinates[current_coords][1];
    // Vérification que les coordonnées sont dans les limites
    return (0 <= new_x && new_x < max_col) && (0 <= new_y && new_y < max_row);
}

/**
 * Fonction appliquant l'effet sur l'image
 * @param data La matrice de pixels de l'image source
 * @param candidate La matrice de pixels de l'image destination (initialement vide)
 * @param rows La hauteur maximale de l'image
 * @param cols La largeur maximale de l'image
 * @param kernel La matrice de facteur de l'effet
 * @param divider Le diviseur des sommes
 * @param offset Le décalage des sommes
 */
 __global__
void device_apply(const uchar* data, uchar* candidate, size_t rows, size_t cols, const int* kernel, float divider, float offset) {

     // Récupération des coordonnées du thread actuel
     uint i = blockIdx.x * blockDim.x + threadIdx.x;
     uint j = blockIdx.y * blockDim.y + threadIdx.y;

     // Vérification que le thread n'est pas hors-limites
     if(i < cols && j < rows) {

         // Initialisation de la somme
         int sum_blue = 0;
         int sum_green = 0;
         int sum_red = 0;

         // Pour chacun des 9 cases dans son voisinage...
         // (size_t provoque des "narrow conversion")
         for (uint current_neighbor_index = 0; current_neighbor_index < device_kernel_size; current_neighbor_index++)

             // Si la case n'est pas hors limite...
             if (device_check(i, j, current_neighbor_index, rows, cols)) {
                 // Récupération du facteur courant (dans le kernel)
                 int current_factor = kernel[current_neighbor_index];

                 // Calcul des coordonnées du pixel à trouver
                 uint new_x = i + device_coordinates[current_neighbor_index][0];
                 uint new_y = j + device_coordinates[current_neighbor_index][1];

                 // Récupération des 3 channels de couleur du pixel
                 uchar blue = data[device_channel_number * (new_y * cols + new_x)];
                 uchar green = data[device_channel_number * (new_y * cols + new_x) + 1];
                 uchar red = data[device_channel_number * (new_y * cols + new_x) + 2];

                 // Ajout dans les sommes des 3 channels
                 sum_blue += blue * current_factor;
                 sum_green += green * current_factor;
                 sum_red = red * current_factor;
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
         // Ajout du channel alpha manuel
         candidate[device_channel_number * (j * cols + i) + 3] = data[device_channel_number * (j * cols + i) + 3];
     }
}

void detection_bord(const_mat_ref mat, uchar* input, uchar* output) {
    //128 threads par blocks
    dim3 thread_size( 32, 4 );
    // Calcule du nombre de block
    dim3 block_size( (( mat.cols - 1) / (thread_size.x - 2) + 1), (( mat.rows - 1 ) / (thread_size.y - 2) + 1) );

    // Instanciation du kernel de l'effet
    std::vector<int> kernel({-1, -1, -1, -1, 8, -1, -1, -1, -1});

    // Pointers du kernel sur le device + allocation
    int* kernel_ptr = nullptr;
    cudaError error_cuda_malloc_kernel_ptr = cudaMalloc(&kernel_ptr, device_kernel_size * sizeof(int));
    if (error_cuda_malloc_kernel_ptr != cudaSuccess) std::cerr << "Error 4 : " << cudaGetErrorString(error_cuda_malloc_kernel_ptr) << std::endl;

    // Copie kernel vers le device
    cudaError error_cuda_mecmcpy_kernel_ptr = cudaMemcpy(kernel_ptr, kernel.data(), device_kernel_size * sizeof(int), cudaMemcpyHostToDevice);
    if (error_cuda_mecmcpy_kernel_ptr != cudaSuccess) std::cerr << "Error 5 : " << cudaGetErrorString(error_cuda_mecmcpy_kernel_ptr) << std::endl;

    // Lancement du processus
    device_apply<<<block_size, thread_size, thread_size.x * thread_size.y>>>(input, output, mat.rows, mat.cols, kernel_ptr, 1.0f, 0.0f);

    // Free du pointer de kernel sur le device
    cudaFree(kernel_ptr);
}

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    // Constant permettant de sotcker la taille necessaires des matrices des images
    const size_t data_size = image.rows * image.cols * device_channel_number;

    // Image vide
    if (image.empty() || !image.data) missing_data();
    // Image qui n'est pas sur  channels
    if (image.channels() != 4) unsupported_channel_number();

    // Pointers de l'image source sur le devide + allocation
    uchar* rgba_data = nullptr;
    cudaError error_cuda_malloc_rgb_data = cudaMalloc(&rgba_data, data_size);
    if (error_cuda_malloc_rgb_data != cudaSuccess) std::cerr << "Error 0 : " << cudaGetErrorString(error_cuda_malloc_rgb_data) << std::endl;

    // Pointers de l'image de retour sur le devide + allocation
    uchar* convolution = nullptr;
    cudaError error_cuda_malloc_convolution = cudaMalloc(&convolution, data_size);
    if (error_cuda_malloc_convolution != cudaSuccess) std::cerr << "Error 1 : " << cudaGetErrorString(error_cuda_malloc_convolution) << std::endl;

    // Copie de l'image source vers le device
    cudaError error_cuda_memcpy_rgba_data = cudaMemcpy(rgba_data, image.data, data_size, cudaMemcpyHostToDevice);
    if (error_cuda_memcpy_rgba_data != cudaSuccess) std::cerr << "Error 2 : " << cudaGetErrorString(error_cuda_memcpy_rgba_data) << std::endl;

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

    // Pointers local de la matrice de pixels de l'image de retour
    auto* output_data = new uchar[data_size];
    // Copie de la matrice de pixels l'image de retour depuis le device vers le locale
    cudaError error_cuda_memcpy_output_data = cudaMemcpy(output_data, convolution, data_size, cudaMemcpyDeviceToHost);
    if (error_cuda_memcpy_output_data != cudaSuccess) std::cerr << "Error 3 : " << cudaGetErrorString(error_cuda_memcpy_output_data) << std::endl;
    // Création de l'image correspondante
    auto result = cv::Mat(image.rows, image.cols, CV_8UC(device_channel_number), output_data);
    // Écriture dans le fichier de sortie
    cv::imwrite("convolution_gpu.png", result);

    // Free les pointers
    delete[] output_data;

    // Free des pointers CUDA
    cudaFree(rgba_data);
    cudaFree(convolution);

    return EXIT_SUCCESS;
}