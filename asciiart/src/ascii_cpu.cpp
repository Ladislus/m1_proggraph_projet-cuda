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
    std::cout << intensity << std::endl;
    return chars[intensity];
}

void process(const cv::Mat& image, std::ofstream& output) {

    for (size_t row = 0; row < image.rows; row++) {
        for (size_t col = 0; col < image.cols; col++) {
            cv::Vec3b intensity = image.at<cv::Vec3b>(row, col);
            output << convert_intensity((intensity[0] + intensity[1] + intensity[2]) % chars.size());
        }
        output << std::endl;
    }
}