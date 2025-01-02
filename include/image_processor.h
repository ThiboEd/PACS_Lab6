#pragma once

#include "gaussian_blur_processor.h"


class ImageProcessor {
public:
    ImageProcessor();
    void loadAndReplicateImage(const char* filename);
    void processImagesWithOpenCL();
    ~ImageProcessor();

private:
    static const int NUM_IMAGES = 5000;
    std::vector<unsigned char> all_images_data;
    std::vector<unsigned char> all_output_data;
    int single_image_size;
    int width, height;
    GaussianBlurProcessor processors[2];  
};
