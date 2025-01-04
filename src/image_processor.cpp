#include "../include/image_processor.h"

using namespace cimg_library;

ImageProcessor::ImageProcessor() : processors{GaussianBlurProcessor(true), GaussianBlurProcessor(false)} {}

ImageProcessor::~ImageProcessor(){}

void ImageProcessor::loadAndReplicateImage(const char* filename){

    CImg<unsigned char> image(filename);

    width = image.width();
    height = image.height();

    single_image_size = width * height;
    all_images_data.resize(single_image_size * NUM_IMAGES);

    for (int i = 0 ; i < NUM_IMAGES ; i++){
        std::copy(image.data(),
                  image.data() + single_image_size,
                  all_images_data.data() + (i * single_image_size));
    }

    std::cout << "Replication has ended :" << std::endl;
    std::cout << "Number of images': " << NUM_IMAGES << std::endl;
    std::cout << "Total size: " << (all_images_data.size() / (1024.0 * 1024.0)) << " MiB" << std::endl;
}

GlobalMetrics ImageProcessor::processImagesWithOpenCL() {
    GlobalMetrics global_metrics = {0};
    cl_platform_id platform;
    cl_device_id devices[2];
    cl_uint num_devices;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, devices, &num_devices);

    
    for (int i = 0; i < 2; i++) {
        processors[i].initializeOpenCL(devices[i]);
        processors[i].printDeviceInfo();
    }

    all_output_data.resize(all_images_data.size());
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_IMAGES; i++) {
        unsigned char* current_input = all_images_data.data() + (i * single_image_size);
        unsigned char* current_output = all_output_data.data() + (i * single_image_size);

        #pragma omp parallel for num_threads(2)
        for (int gpu = 0; gpu < 2; gpu++) {
            ProcessingMetrics metrics = processors[gpu].processImage(
                current_input,
                current_output,
                width,
                height
            );

            #pragma omp critical
            {
                global_metrics.total_memory_transfer_time += metrics.memory_transfer_time;
                global_metrics.total_kernel_execution_time += metrics.kernel_execution_time;
                global_metrics.peak_memory_usage = std::max(global_metrics.peak_memory_usage, metrics.memory_used);
                global_metrics.avg_gpu_occupancy[gpu] += metrics.gpu_occupancy;
            }
        }

        if (i % 100 == 0) {
            std::cout << "Processed " << i << " images..." << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    global_metrics.total_processing_time = 
        std::chrono::duration<double>(end_time - start_time).count();
    global_metrics.avg_time_per_image = global_metrics.total_processing_time / NUM_IMAGES;

    for (int gpu = 0; gpu < 2; gpu++) {
        global_metrics.avg_gpu_occupancy[gpu] /= NUM_IMAGES;
    }

    return global_metrics;
}

void ImageProcessor::printMetrics(const GlobalMetrics& metrics) {
    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Total processing time: " << metrics.total_processing_time << " seconds" << std::endl;
    std::cout << "Average time per image: " << metrics.avg_time_per_image << " seconds" << std::endl;
    std::cout << "Total memory transfer time: " << metrics.total_memory_transfer_time << " seconds" << std::endl;
    std::cout << "Total kernel execution time: " << metrics.total_kernel_execution_time << " seconds" << std::endl;
    std::cout << "Peak memory usage: " << (metrics.peak_memory_usage / (1024*1024)) << " MB" << std::endl;
    std::cout << "Average GPU occupancy: GPU0: " << metrics.avg_gpu_occupancy[0] 
              << "%, GPU1: " << metrics.avg_gpu_occupancy[1] << "%" << std::endl;
}


