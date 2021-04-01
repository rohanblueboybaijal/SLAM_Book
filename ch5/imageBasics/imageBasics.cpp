#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
    cv::Mat image;
    image = cv::imread(argv[1]);

    if(image.data == nullptr) {
        std::cerr << "file "<< argv[1] << " does not exist." <<std::endl;
        return 0;
    }

    std::cout << "Image cols: " << image.cols << ", rows: " << image.rows << ", channel: " << image.channels() << std::endl;

    cv::imshow("image", image);
    cv::waitKey(0);

    if(image.type() != CV_8UC1 && image.type() != CV_8UC3 ) {
        std::cout << "image type incorrect" <<std::endl;
        return 0;
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for(size_t y = 0; y < image.rows; y++) {
        unsigned char *row_ptr = image.ptr<unsigned char>(y); 
        // row_ptr is the pointer to the yth row
        for(size_t x = 0; x < image.cols; x++) {
            unsigned char *data_ptr = &row_ptr[x * image.channels()]; //pointer to I(x,y)

            for(int c = 0; c != image.channels(); c++) {
                unsigned char data =  data_ptr[c];
                // pixel of I(x,y) in c-th channel
            }
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> (t2 - t1);

    std::cout << "time used: " << time_used.count() << " seconds." << std::endl;

    // copying cv::Mat
    // just makes a reference => changing another_image will also reflect in the original image
    cv::Mat another_image = image;
    another_image(cv::Rect(0,0,100,100)).setTo(0); // set top-left 100*100 block to 0

    cv::imshow("image", image);
    cv::waitKey(0);

    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0,0,100,100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}