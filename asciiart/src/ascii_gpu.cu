#include "ascii.hpp"

__constant__ const size_t device_size = 71;
__constant__ const uchar device_chars[71] = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
__constant__ const float device_divider = 255.0f / static_cast<float>(device_size - 1);

/**
 * Fonction permettant d'obtenir le caractère ASCII correspondant à l'intensité donnée
 * @param intensity L'intensité du caractère à convertie
 * @return Le caractère ASCII correspondant
 */
 __device__
uchar device_convert_intensity(uchar intensity) {
    // Convertion de l'intensité en indice dans le set de caractère
    int rounded = static_cast<int>(static_cast<float>(intensity) / device_divider);
    // Vérification que l'indice n'est pas OOB
    assert(rounded < device_size);
    // Retourne le caractère correspondant
    return device_chars[rounded];
}

/**
 * Fonction de transformation de l'image d'entrée en ASCII
 * @param image L'image source
 * @param output Le stream vers le fichier de sortie
 */
__global__
void asciify(const uchar* data, uchar* candidate, size_t rows, size_t cols) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < cols && j < rows)
        candidate[j * cols + i] = device_convert_intensity(data[j * cols + i]);
}

int main(int argc, char** argv) {

    // Chemin vers le fichier source manquant
    if (argc != 2) missing_argument();

    // Récupération de l'image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    size_t data_size = image.rows * image.cols;

    // Image vide
    if (image.empty() || !image.data) missing_data();

    auto* output_data = new uchar[data_size];
    for (size_t i = 0; i < data_size; i++) output_data[i] = 255;

    uchar* grayscaled;
    uchar* asciified;
    cudaError e0 = cudaMalloc(&grayscaled, data_size);
    if (e0 != cudaSuccess) std::cerr << "Error 0 : " << cudaGetErrorString(e0) << std::endl;
    cudaError e1 = cudaMalloc(&asciified, data_size);
    if (e1 != cudaSuccess) std::cerr << "Error 1 : " << cudaGetErrorString(e1) << std::endl;

    cudaError e2 = cudaMemcpy(grayscaled, image.data, data_size, cudaMemcpyHostToDevice);
    if (e2 != cudaSuccess) std::cerr << "Error 2 : " << cudaGetErrorString(e2) << std::endl;

    // TIMERS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 thread_size( 32, 4 ); //128 threads
    dim3 block_size( (( image.cols - 1) / (thread_size.x - 2) + 1), (( image.rows - 1 ) / (thread_size.y - 2) + 1) );
    asciify<<<block_size, thread_size, thread_size.x * thread_size.y>>>(grayscaled, asciified, image.rows, image.cols);

    // TIMER
    cudaEventRecord(stop);
    cudaEventSynchronize( stop );
    float duration;
    cudaEventElapsedTime( &duration, start, stop );
    std::cout << "Processing took: " << duration << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError e3 = cudaMemcpy(output_data, asciified, data_size, cudaMemcpyDeviceToHost);
    if (e3 != cudaSuccess) std::cerr << "Error 3 : " << cudaGetErrorString(e3) << std::endl;

    // Ouverture du stream vers le fichier de sortie
    std::ofstream output("ascii_gpu.txt");
    size_t i, j;
    for (i = 0; i < image.rows; i++) {
        for (j = 0; j < image.cols; j++) {
            output << output_data[i * image.cols + j] << output_data[i * image.cols + j];
            char x = convert_intensity(image.at<uchar>(i, j));
            if (x != output_data[i * image.cols + j]) {
                std::clog << "Device: " << output_data[i * image.cols + j] << "  ;  Expected: " << x << std::endl;
            }
        }
        output << std::endl;
    }
    output.close();

    // Free les pointers
    delete[] output_data;

    cudaFree(grayscaled);
    cudaFree(asciified);

    return EXIT_SUCCESS;
}