#define cimg_use_jpeg
#define CL_TARGET_OPENCL_VERSION 120

#include <iostream>
#include "CImg.h"
#include "cl.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <chrono>

#define DIM 3 
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

using namespace cimg_library;

std::vector<float> create_gaussian_matrix(double sigma = 1) {
    /*
    Compute the convolution matrix for all values of sigma (1 by default)
    and a mean of 0 with the desired filter size (here 3).
    */
    
    double kernel[DIM][DIM];
    std::vector<float> result;
    double K = 1 / (2 * M_PI * pow(sigma, 2));
    double sum = 0.0;

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            kernel[i][j] = K * exp(-((pow(i - (DIM / 2), 2) + pow(j - (DIM / 2), 2)) / (2 * sigma * sigma)));
            sum += kernel[i][j];
        }
    }
    // Normalize the kernel
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            kernel[i][j] /= sum;
            result.push_back(static_cast<float>(kernel[i][j])); //Not sure we need static_cast here. 
        }
    }
    return result;
}

void check_error(cl_int err, const char *operation){
    if (err != CL_SUCCESS){
        fprintf(stderr , "Error during operation '%s',",operation);
        exit(EXIT_FAILURE);
    }
}

int main() {

    std::chrono::time_point<std::chrono::high_resolution_clock> overall_start_time, overall_end_time;
    overall_start_time = std::chrono::high_resolution_clock::now();

    CImg<unsigned char> image("image2.jpg");

    int width = image.width();
    int height = image.height();

    std::vector<float> gaussian_kernel = create_gaussian_matrix(1.0);

    std::vector<unsigned char> img_data (image.data() , image.data() + width * height);
    std::vector<unsigned char> output_data (width * height);

    std::vector <int> dim = {height, width};

    cl_int err;

    cl_platform_id platform;
    err = clGetPlatformIDs(1 , &platform,NULL);
    check_error(err, "Finding platform");

    cl_device_id device;
    err = clGetDeviceIDs(platform, DEVICE, 1,&device,NULL);
    check_error(err, "Finding device");

    cl_context context = clCreateContext(NULL , 1 , &device , NULL , NULL , &err);
    check_error(err, "Creating context");

    cl_command_queue commands = clCreateCommandQueue(context , device , CL_QUEUE_PROFILING_ENABLE , &err);
    check_error(err , "Failed to create queue");

    FILE *fileHandler = fopen("gaussian_kernel.cl", "r");
    fseek (fileHandler, 0 , SEEK_END);
    size_t fileSize = ftell(fileHandler);
    rewind(fileHandler);

    char* sourceCode = (char*) malloc(fileSize + 1);
    sourceCode [fileSize] = '\0';
    fread(sourceCode , sizeof(char), fileSize, fileHandler);
    fclose(fileHandler);

    cl_program program = clCreateProgramWithSource(context,1,(const char**)&sourceCode, &fileSize, &err);
    check_error(err , "Failed to create program with source");
    free(sourceCode);

    err = clBuildProgram(program, 0 , NULL , NULL , NULL , NULL);
    if(err != CL_SUCCESS){
    size_t len;
    char buffer[2048];
    printf("Error: Some error at building process.\n");
    clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            sizeof(buffer),
                            buffer,
                            &len);
    printf("%s\n",buffer);
    exit(-1);
    }

    cl_kernel kernel = clCreateKernel(program , "gaussian_blur" , &err);
    check_error(err , "Failed to create kernel from the program.");

    //Creating all buffers for OpenCL memory
    size_t buffer_size = height * width * sizeof(unsigned char);
    size_t gaussian_buffer_size = gaussian_kernel.size() * sizeof(float);
    size_t dim_buffer_size = dim.size() * sizeof(int);
    cl_mem input_buffer = clCreateBuffer(context,
                                         CL_MEM_READ_ONLY,
                                         buffer_size,
                                         NULL,
                                         &err);
    check_error(err, "Creating input buffer");
    //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, I could have used this instead of
    //CL_MEM_READ_ONLY, I did so to compute the bandwidth. 

    //To compute bandwidth from host to device. 
    cl_event write_event;
    cl_ulong write_start, write_end;
    err = clEnqueueWriteBuffer(commands,
                               input_buffer,
                               CL_TRUE,
                               0,
                               buffer_size,
                               img_data.data(),
                               0,
                               NULL,
                               &write_event);
    check_error(err ,"Writing to device buffer");
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write_end, NULL);
    double write_time = (write_end - write_start) / 1e6;
    printf("Host to Device Bandwidth: %.4f MB/s\n", (buffer_size / (1024.0 * 1024.0)) / (write_time / 1000.0));

    cl_mem output_buffer = clCreateBuffer(context,
                                          CL_MEM_WRITE_ONLY,
                                          buffer_size,
                                          NULL,
                                          &err);
    check_error(err, "Creating output buffer");
    cl_mem kernel_buffer = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          gaussian_buffer_size,
                                          gaussian_kernel.data(),
                                          &err);
    check_error(err, "Creating kernel buffer");
    cl_mem dim_buffer = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       dim_buffer_size,
                                       dim.data(),
                                       &err);
    check_error(err, "Creating kernel buffer");

    //Configuring arguments of kernel, we have 4 of them

    err = clSetKernelArg(kernel , 0 , sizeof(cl_mem), &input_buffer);
    check_error(err, "Setting input buffer argement");
    err = clSetKernelArg(kernel , 1 , sizeof(cl_mem), &output_buffer);
    check_error(err, "Setting output buffer argement");
    err = clSetKernelArg(kernel , 2 , sizeof(cl_mem), &kernel_buffer);
    check_error(err, "Setting kernel buffer argement");
    err = clSetKernelArg(kernel , 3 , sizeof(cl_mem), &dim_buffer);
    check_error(err, "Setting dimension buffer argement");

    //Launch kernel
    size_t global_size[2] = {static_cast<size_t>(width) , static_cast<size_t>(height)};
    cl_event kernel_event;
    err = clEnqueueNDRangeKernel(commands,
                                kernel,
                                2,
                                NULL,
                                global_size,
                                NULL,
                                0,
                                NULL,
                                &kernel_event);
    check_error(err, "Executing kernel");
    clWaitForEvents(1,&kernel_event);


    //Compute bandwitdh from device to host
    cl_event read_event;
    err = clEnqueueReadBuffer(commands,
                              output_buffer,
                              CL_TRUE,
                              0,
                              buffer_size,
                              output_data.data(),
                              0,
                              NULL,
                              &read_event);
    check_error(err, "Reading output buffer");
    cl_ulong read_start, read_end;
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_end, NULL);
    double read_time = (read_end - read_start) / 1e6; // Convert to milliseconds
    printf("Device to Host Bandwidth: %.4f MB/s\n", (buffer_size / (1024.0 * 1024.0)) / (read_time / 1000.0));

    clFinish(commands);

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double kernel_execution_time = time_end - time_start;
    printf("Kernel Execution time is: %0.4f milliseconds \n",kernel_execution_time / 1000000.0);

    size_t totlal_operations = width * height * DIM * DIM;
    double throughput = totlal_operations / (kernel_execution_time / 1e9);
    printf("Kernel Throughput: %.4f GFLOP/s\n", throughput / 1e9); // Diviser par 1e9 pour exprimer en GFLOP/s

    size_t total_transfer_size = buffer_size * 2 + gaussian_buffer_size + dim_buffer_size;
    printf("Kernel Memory Bandwidth: %.4f MB/s\n", (total_transfer_size / (1024.0 * 1024.0)) / (kernel_execution_time / (1000.0 * 1e6))); // *1e6 to convert into milliseconds 1024 * 1024 to convert bytes to MB in base 2. 

    size_t opencl_memory_footprint = 0;
    opencl_memory_footprint = buffer_size * 2 + gaussian_buffer_size + dim_buffer_size;

    size_t host_memory_footprint = 0;
    host_memory_footprint += img_data.size() * sizeof(unsigned char);
    host_memory_footprint += output_data.size() * sizeof(unsigned char);
    host_memory_footprint += gaussian_kernel.size() * sizeof(float);
    host_memory_footprint += dim.size() * sizeof(int);

    size_t total_memory_footprint = opencl_memory_footprint + host_memory_footprint;

    printf("OpenCL Memory Footprint: %.4f MB\n", opencl_memory_footprint / (1024.0 * 1024.0));
    printf("Host Memory Footprint: %.4f MB\n", host_memory_footprint / (1024.0 * 1024.0));
    printf("Total Memory Footprint: %.4f MB\n", total_memory_footprint / (1024.0 * 1024.0));

    CImg<unsigned char> output_image(output_data.data(), width, height,1,1);
    output_image.save("output_blurred2.jpg");

    clReleaseEvent(read_event);
    clReleaseEvent(write_event);
    clReleaseEvent(kernel_event);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(kernel_buffer);
    clReleaseMemObject(dim_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    overall_end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time).count();
    printf("Total execution time : %.4f ms\n", total_time);
    
    return 0;
}
