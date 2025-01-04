#pragma once

#include "gaussian_blur_processor.h"
#include <chrono>

struct GlobalMetrics {
    double total_processing_time;
    double avg_time_per_image;
    double total_memory_transfer_time;
    double total_kernel_execution_time;
    size_t peak_memory_usage;
    double avg_gpu_occupancy[2];
};

class ImageProcessor {
public:
    ImageProcessor();
    void loadAndReplicateImage(const char* filename);
    GlobalMetrics processImagesWithOpenCL();
    void printMetrics(const GlobalMetrics& metrics);
    ~ImageProcessor();

private:
    static const int NUM_IMAGES = 1000;
    std::vector<unsigned char> all_images_data;
    std::vector<unsigned char> all_output_data;
    int single_image_size;
    int width, height;
    GaussianBlurProcessor processors[2];
};

