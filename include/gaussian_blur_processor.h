#pragma once

#include <CL/cl.h>
#include <vector>
#include <CImg.h>
#include <iostream>

class GaussianBlurProcessor {
    public:
        GaussianBlurProcessor(bool boolean);
        void initializeOpenCL(cl_device_id device);
        void processImage(const unsigned char* input_data, unsigned char* output_data,
            int width , int height);
        ~GaussianBlurProcessor();

    private:
        cl_context context;
        cl_command_queue commands;
        cl_program program;
        cl_kernel kernel;
        std::vector<float> gaussian_kernel;
        bool first_gpu;

        void check_error(cl_int err, const char* operation);
        std::vector<float> create_gaussian_matrix (double sigma = 1);
};
