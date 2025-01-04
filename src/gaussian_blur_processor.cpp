#include "../include/gaussian_blur_processor.h"
#include <math.h>
#include <chrono>

#define DIM 3 

GaussianBlurProcessor::GaussianBlurProcessor(bool boolean){
    gaussian_kernel = create_gaussian_matrix(1.0);
    first_gpu = boolean;
} 

GaussianBlurProcessor::~GaussianBlurProcessor(){
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram (program);
    if (commands) clReleaseCommandQueue (commands);
    if (context) clReleaseContext (context);
}

void GaussianBlurProcessor::check_error(cl_int err, const char *operation){
        if (err != CL_SUCCESS){
        fprintf(stderr , "Error during operation '%s',",operation);
        exit(EXIT_FAILURE);
    }
}

void GaussianBlurProcessor::initializeOpenCL(cl_device_id device) {
    cl_int err;
    this->device = device;  


    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "Creating context");


    commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_error(err, "Creating command queue");


    FILE *fileHandler = fopen("gaussian_kernel.cl", "r");
    if (!fileHandler) {
        fprintf(stderr, "Failed to load kernel file\n");
        exit(1);
    }
    
    fseek(fileHandler, 0, SEEK_END);
    size_t fileSize = ftell(fileHandler);
    rewind(fileHandler);

    char* sourceCode = (char*)malloc(fileSize + 1);
    sourceCode[fileSize] = '\0';
    fread(sourceCode, sizeof(char), fileSize, fileHandler);
    fclose(fileHandler);


    program = clCreateProgramWithSource(context, 1, (const char**)&sourceCode, &fileSize, &err);
    check_error(err, "Creating program");
    free(sourceCode);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("Build error: %s\n", buffer);
        exit(-1);
    }


    kernel = clCreateKernel(program, "gaussian_blur", &err);
    check_error(err, "Creating kernel");
}

std::vector<float> GaussianBlurProcessor::create_gaussian_matrix(double sigma){
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

void GaussianBlurProcessor::printDeviceInfo() {
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    cl_uint compute_units;
    char device_name[128];
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

    std::cout << "Device: " << device_name << std::endl;
    std::cout << "Global Memory: " << (global_mem_size / (1024*1024)) << " MB" << std::endl;
    std::cout << "Local Memory: " << (local_mem_size / 1024) << " KB" << std::endl;
    std::cout << "Compute Units: " << compute_units << std::endl;
}

double GaussianBlurProcessor::getEventExecutionTime(cl_event event) {
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    return (time_end - time_start) / 1.0e9;
}

double GaussianBlurProcessor::calculateGPUOccupancy(size_t global_work_items, size_t local_work_items) {
    cl_uint compute_units;
    size_t max_work_group_size;
    cl_uint max_work_items_per_dim[3];
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_items_per_dim), max_work_items_per_dim, NULL);
    

    size_t max_concurrent_workgroups = compute_units;

    size_t total_workgroups = (global_work_items + local_work_items - 1) / local_work_items;

    double occupancy = (static_cast<double>(total_workgroups) / max_concurrent_workgroups);
    return std::min(occupancy * 100.0, 100.0);
}


ProcessingMetrics GaussianBlurProcessor::processImage(const unsigned char* input_data, 
                                                    unsigned char* output_data,
                                                    int width, int height) {
    ProcessingMetrics metrics = {};  
    cl_event write_event, kernel_event, read_event;
    cl_int err;

    int GPU_height = height / 2;
    int remainder = height % 2;

    int start_height, current_height;
    if (first_gpu) {
        start_height = 0;
        current_height = GPU_height;
    } else {
        start_height = GPU_height;
        current_height = GPU_height + remainder;
    }
    
    std::vector<int> dim = {current_height, width};


    size_t buffer_size = current_height * width * sizeof(unsigned char);
    size_t gaussian_buffer_size = gaussian_kernel.size() * sizeof(float);
    size_t dim_buffer_size = dim.size() * sizeof(int);
    
    metrics.memory_used = buffer_size * 2 + // input et output buffers
                         gaussian_buffer_size + // kernel gaussien
                         dim_buffer_size;  // buffer des dimensions


    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    check_error(err, "Creating input buffer");
    
    cl_mem kernel_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                        gaussian_buffer_size, gaussian_kernel.data(), &err);
    check_error(err, "Creating kernel buffer");
    
    cl_mem dim_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                     dim_buffer_size, dim.data(), &err);
    check_error(err, "Creating dim buffer");
    
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, NULL, &err);
    check_error(err, "Creating output buffer");

    auto cpu_start = std::chrono::high_resolution_clock::now();

    err = clEnqueueWriteBuffer(commands, input_buffer, CL_FALSE, 0, buffer_size,
                              input_data + (start_height * width), 0, NULL, &write_event);
    check_error(err, "Enqueuing write buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &kernel_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dim_buffer);
    check_error(err, "Setting kernel arguments");

    size_t max_work_group_size;
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, 
                                  sizeof(max_work_group_size), &max_work_group_size, NULL);
    check_error(err, "Getting work group info");

    size_t global_size[2] = {static_cast<size_t>(width), static_cast<size_t>(current_height)};
    size_t local_size[2] = {16, 16};

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global_size, local_size,
                                1, &write_event, &kernel_event);
    check_error(err, "Enqueuing kernel");

    err = clEnqueueReadBuffer(commands, output_buffer, CL_TRUE, 0, buffer_size,
                             output_data + (start_height * width), 
                             1, &kernel_event, &read_event);
    check_error(err, "Reading output buffer");

    clFinish(commands);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    metrics.memory_transfer_time = getEventExecutionTime(write_event) + getEventExecutionTime(read_event);
    metrics.kernel_execution_time = getEventExecutionTime(kernel_event);
    
    metrics.total_processing_time = std::chrono::duration<double>(cpu_end - cpu_start).count();
    
    metrics.gpu_occupancy = calculateGPUOccupancy(global_size[0] * global_size[1], 
                                                local_size[0] * local_size[1]);

    metrics.overhead_time = metrics.total_processing_time - 
                          (metrics.memory_transfer_time + metrics.kernel_execution_time);

    clReleaseEvent(write_event);
    clReleaseEvent(kernel_event);
    clReleaseEvent(read_event);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(kernel_buffer);
    clReleaseMemObject(dim_buffer);

    return metrics;
}



