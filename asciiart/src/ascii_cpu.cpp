#include "ascii_cpu.hpp"

int main(int argc, char** argv) {

    if (argc != 2) missing_argument();

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    if (!image.data) missing_data();

    std::ofstream output("out.txt");
    process(image, output);

    output.close();

    return EXIT_SUCCESS;
}

void missing_argument() {
    printf("usage: DisplayImage.out <Image_Path>\n");
    exit(EXIT_FAILURE);
}

void missing_data() {
    printf("No image data \n");
    exit(EXIT_FAILURE);
}

char convert_intensity(uchar intensity) {
    int rounded = static_cast<int>(static_cast<float>(intensity) / divider);
    std::cout << rounded << std::endl;
    assert(rounded < chars.size());
    return chars[rounded];
}

void process(const cv::Mat& image, std::ofstream& output) {

    for (size_t row = 0; row < image.rows; row++) {
        for (size_t col = 0; col < image.cols; col++) {
            uchar converted_char = convert_intensity(image.at<uchar>(row, col));
            output << converted_char << converted_char << converted_char;
        }
        output << std::endl;
    }
}