#include "ascii_cpu.hpp"

int main(int argc, char** argv) {

    if (argc != 2) missing_argument();

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    if (!image.data) missing_data();

    cv::Mat new_image = *process(image);

    namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    imshow("Display Image", new_image);
    cv::waitKey(0);

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
    return chars[intensity];
}

cv::Mat* process(const cv::Mat& image) {
    cv::Mat* candidate = new cv::Mat(image.rows, image.cols, CV_8UC1);

    for (size_t row = 0; row < image.rows; row++) {
        for (size_t col = 0; col < image.cols; col++) {
            cv::Vec3b intensity = image.at<cv::Vec3b>(row, col);

            candidate->at<uchar>(row, col) = convert_intensity((intensity[0] + intensity[1] + intensity[2]) * chars.size());
        }
    }

    return candidate;
}