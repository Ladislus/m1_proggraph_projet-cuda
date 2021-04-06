#include "ascii.hpp"

// Stockage de la taille du set de caractères en constant sur le device
__constant__ const size_t device_size = 70;
// Stockage du set de caractères en constant sur le device (70 caractères + caractère de fin de string)
__constant__ const uchar device_chars[device_size + 1] = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
// Stockage de la constant de division sur le device
__constant__ const float device_divider = 255.0f / static_cast<float>(device_size - 1);

/**
 * Fonction GPU permettant d'obtenir le caractère ASCII correspondant à l'intensité donnée
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
 * Fonction GPU appliquant l'effet ASCII sur l'image
 * @param data L'image source
 * @param candidate L'image de retour
 * @param rows Le nombre de lignes
 * @param cols Le nombre de colonnes
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

    // Pointers de l'image source sur le devide + allocation
    uchar* grayscaled;
    cudaError e0 = cudaMalloc(&grayscaled, data_size);
    if (e0 != cudaSuccess) std::cerr << "Error 0 : " << cudaGetErrorString(e0) << std::endl;
    // Pointers de l'image de retour sur le devide + allocation
    uchar* asciified;
    cudaError e1 = cudaMalloc(&asciified, data_size);
    if (e1 != cudaSuccess) std::cerr << "Error 1 : " << cudaGetErrorString(e1) << std::endl;

    // Copie de l'image source vers le device
    cudaError e2 = cudaMemcpy(grayscaled, image.data, data_size, cudaMemcpyHostToDevice);
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
    asciify<<<block_size, thread_size, thread_size.x * thread_size.y>>>(grayscaled, asciified, image.rows, image.cols);

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
    cudaError e3 = cudaMemcpy(output_data, asciified, data_size, cudaMemcpyDeviceToHost);
    if (e3 != cudaSuccess) std::cerr << "Error 3 : " << cudaGetErrorString(e3) << std::endl;

    // Ouverture du stream vers le fichier de sortie
    std::ofstream output("ascii_gpu.txt");
    // Écriture de l'image de retour vers le fichier texte de sortie
    for (size_t i = 0; i < image.rows; i++) {
        for (size_t j = 0; j < image.cols; j++)
            output << output_data[i * image.cols + j] << output_data[i * image.cols + j] << output_data[i * image.cols + j];
        output << std::endl;
    }
    // Fermeture du stream vers le fichier de sortie
    output.close();

    // Free les pointers
    delete[] output_data;

    // Free des pointers CUDA
    cudaFree(grayscaled);
    cudaFree(asciified);

    return EXIT_SUCCESS;
}