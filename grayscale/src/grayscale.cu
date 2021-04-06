#include "grayscale.hpp"

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
__global__
void grayscale(const uchar* data, uchar* candidate, size_t rows, size_t cols) {

    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if( i < cols && j < rows )
        candidate[ j * cols + i ] = (307 * data[3 * (j * cols + i)] + 604 * data[3 * (j * cols + i ) + 1] + 113 * data[3 * ( j * cols + i ) + 2]) >> 10; // >>10 <=> division par 1024
}

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    size_t data_size = image.rows * image.cols * 3;

    // Image vide
    if (image.empty() || !image.data) missing_data();

    auto* output_data = new uchar[data_size];

    uchar* rgb;
    uchar* grayscaled;
    cudaError e0 = cudaMalloc(&rgb, data_size);
    if (e0 != cudaSuccess) std::cerr << "Error 0 : " << cudaGetErrorString(e0) << std::endl;
    cudaError e1 = cudaMalloc(&grayscaled, data_size);
    if (e1 != cudaSuccess) std::cerr << "Error 1 : " << cudaGetErrorString(e1) << std::endl;

    cudaError e2 = cudaMemcpy(rgb, image.data, data_size, cudaMemcpyHostToDevice);
    if (e2 != cudaSuccess) std::cerr << "Error 2 : " << cudaGetErrorString(e2) << std::endl;

    // TIMERS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 thread_size( 32, 4 ); //128 threads
    dim3 block_size( (( image.cols - 1) / (thread_size.x - 2) + 1), (( image.rows - 1 ) / (thread_size.y - 2) + 1) );
    grayscale<<<block_size, thread_size, thread_size.x * thread_size.y>>>(rgb, grayscaled, image.rows, image.cols);

    // TIMER
    cudaEventRecord(stop);
    cudaEventSynchronize( stop );
    float duration;
    cudaEventElapsedTime( &duration, start, stop );
    std::cout << "time=" << duration << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError e3 = cudaMemcpy(output_data, grayscaled, data_size, cudaMemcpyDeviceToHost);
    if (e3 != cudaSuccess) std::cerr << "Error 3 : " << cudaGetErrorString(e3) << std::endl;
    auto* result = new cv::Mat(image.rows, image.cols, CV_8UC1, output_data);
    cv::imwrite("grayscale_gpu.png", *result);

    // Free les pointers
    delete[] output_data;
    delete result;

    cudaFree(rgb);
    cudaFree(grayscaled);

    return EXIT_SUCCESS;
}
