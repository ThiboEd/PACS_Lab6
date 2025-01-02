#include "../include/gaussian_blur_processor.h"

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

GaussianBlurProcessor::initializeOpenCL(cl_device device){

    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    context = clCreateContext(NULL , 1 , &device , NULL, NULL , &err);
    check_error(err, "Creating context");

    commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_error(err, "Creating command queue");

    FILE *fileHandler = fopen("gaussian_kernel.cl","r");
    fseek(fileHandler,0,SEEK_END);
    size_t fileSize = ftell(fileHandler);
    rewind(fileHandler);

    char* sourceCode =(char*)malloc(fileSize + 1);
    sourceCode[fileSize] = '\0';
    fread(sourceCode, sizeof(char), fileSize, fileHandler);
    fclose(fileHandler);

    program = clCreateProgramWithSource(context, 1 , (const char**)&sourceCode , &fileSize , &err);
    check_error(err , "Creating program");
    free(sourceCode);

    err = clBuildProgram(program,0,NULL,NULL,NULL,NULL);
    if (err !=CL_SUCCESS){
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer),buffer,&len);
        printf("%s\n",buffer);
        exit(-1);
    }

    kernel = clCreateKernel(program , "gaussian_blur" , &err);
    check_error(err , "Creating kernel");
}

GaussianBlurProcessor::processImage(const unsigned char* input_data , unsigned char* output_data,
                                    int width , int height){

    cl_mem input_buffer , kernel_buffer , dim_buffer , output_buffer;

    int GPU_height = height / 2;
    int remainder = height % 2;

    int start_height , current_height;

    if (first_gpu){
        start_height = 0;
        current_height = GPU_height;
    }
    else{
        start_height = GPU_height;
        current_height = GPU_height + remainder;
    }

    std::vector<int> dim = {current_height , width};

    size_t buffer_size = current_height * width * sizeof(unsigned char);
    size_t gaussian_buffer_size = gaussian_kernel.size() * sizeof(float);
    size_t dim_buffer_size = dim.size() * sizeof(int);

    input_buffer = clCreateBuffer(context , CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    check_error(err, "Creating input buffer");

    kernel_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gaussian_buffer_size, gaussian_kernel.data(),&err);
    check_error(err, "Creating kernel buffer");

    dim_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim_buffer_size,dim.data(),err);
    check_error(err, "Creating dim buffer");

    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size,NULL , &err);
    check_error(err, "Creating output buffer");

    err = clEnqueueWriteBuffer(commands,
                               input_buffer,
                               Cl_TRUE,
                               0,
                               buffer_size,
                               input_data + (start_height * width),
                               0, NULL , NULL);
    check_error(err, "Writing to input buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1 , sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &kernel_buffer);
    err |= clSetKernelArg(kernel , 3, sizeof(cl_mem), &dim_buffer);
    check_error(err , "Setting kernel arguments");

    size_t global_size[2] = {static_cast<size_t>(width) , static_cast<size_t>(current_height)};
    err= clEnqueueNDRangeKernel(commands,
                                kernel,
                                2,
                                NULL,
                                global_size,
                                NULL,
                                0, NULL, NULL);
    check_error(err,"Executing kernel");

    err = clEnqueueReadBuffer(commands,
                              output_buffer,
                              CL_TRUE,
                              0,
                              buffer_size,
                              output_data + (start_height * width),
                              0, NULL, NULL);
    check_error(err , "Reading output buffer");

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(kernel_buffer);
    clReleaseMemObject(dim_buffer);
}


int main(){

}