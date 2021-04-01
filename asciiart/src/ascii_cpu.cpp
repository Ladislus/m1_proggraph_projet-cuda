#include <opencv2/opencv.hpp>
#include <cstdlib>

#include "ascii_cpu.hpp"

int main(int argc, char** argv ) {

    if ( argc != 2 ) missing_argument();

    cv::Mat image = cv::imread( argv[1], 1 );

    if ( !image.data ) missing_data();

    namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    imshow("Display Image", image);
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