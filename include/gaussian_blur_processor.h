#pragma once
#define cimg_use_jpeg

#include <CL/cl.h>
#include <vector>
#include <CImg.h>
#include <iostream>

struct ProcessingMetrics {
    double memory_transfer_time;    
    double kernel_execution_time;   
    double total_processing_time;  
    double overhead_time;          
    size_t memory_used;            
    double gpu_occupancy;          
    double transfer_bandwidth;      
};

class GaussianBlurProcessor {
    public:
        GaussianBlurProcessor(bool boolean);
        void initializeOpenCL(cl_device_id device);
        ProcessingMetrics processImage(const unsigned char* input_data, unsigned char* output_data,
            int width, int height);
        void printDeviceInfo();
        ~GaussianBlurProcessor();

    private:
        cl_context context;
        cl_command_queue commands;
        cl_program program;
        cl_kernel kernel;
        cl_device_id device;
        std::vector<float> gaussian_kernel;
        bool first_gpu;

        void check_error(cl_int err, const char* operation);
        std::vector<float> create_gaussian_matrix(double sigma = 1);
        double getEventExecutionTime(cl_event event);
        double calculateGPUOccupancy(size_t global_work_items, size_t local_work_items);
};
