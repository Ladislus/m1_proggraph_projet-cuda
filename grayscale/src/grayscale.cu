#include "grayscale.hpp"

/**
 * Fonction GPU appliquant l'effet grayscale sur l'image
 * @param data L'image source
 * @param candidate L'image de retour
 * @param rows Le nombre de lignes
 * @param cols Le nombre de colonnes
 */
__global__
void grayscale(const uchar* data, uchar* candidate, size_t rows, size_t cols) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < cols && j < rows)
        candidate[j * cols + i] = (307 * data[3 * (j * cols + i)] + 604 * data[3 * (j * cols + i ) + 1] + 113 * data[3 * ( j * cols + i ) + 2]) >> 10; // >>10 <=> division par 1024
}

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    size_t data_size = image.rows * image.cols * 3;

    // Image vide
    if (image.empty() || !image.data) missing_data();

    // Pointers de l'image source sur le devide + allocation
    uchar* rgb;
    cudaError e0 = cudaMalloc(&rgb, data_size);
    if (e0 != cudaSuccess) std::cerr << "Error 0 : " << cudaGetErrorString(e0) << std::endl;
    // Pointers de l'image de retour sur le devide + allocation
    uchar* grayscaled;
    cudaError e1 = cudaMalloc(&grayscaled, data_size);
    if (e1 != cudaSuccess) std::cerr << "Error 1 : " << cudaGetErrorString(e1) << std::endl;

    // Copie de l'image source vers le device
    cudaError e2 = cudaMemcpy(rgb, image.data, data_size, cudaMemcpyHostToDevice);
    if (e2 != cudaSuccess) std::cerr << "Error 2 : " << cudaGetErrorString(e2) << std::endl;

    // TIMERS avant
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //128 threads par blocks
    dim3 thread_size( 32, 4 );
    // Calcule du nombre de block
    dim3 block_size( (( image.cols - 1) / (thread_size.x - 2) + 1), (( image.rows - 1 ) / (thread_size.y - 2) + 1) );
    // Lancement du calcul
    grayscale<<<block_size, thread_size, thread_size.x * thread_size.y>>>(rgb, grayscaled, image.rows, image.cols);

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
    cudaError e3 = cudaMemcpy(output_data, grayscaled, data_size, cudaMemcpyDeviceToHost);
    if (e3 != cudaSuccess) std::cerr << "Error 3 : " << cudaGetErrorString(e3) << std::endl;
    // Création de l'image correspondante
    auto result =cv::Mat(image.rows, image.cols, CV_8UC1, output_data);
    // Écriture dans le fichier de sortie
    cv::imwrite("grayscale_gpu.png", result);

    // Free les pointers
    delete[] output_data;

    // Free des pointers CUDA
    cudaFree(rgb);
    cudaFree(grayscaled);

    return EXIT_SUCCESS;
}
