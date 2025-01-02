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

void ImageProcessor::processImagesWithOpenCL(){

    cl_platform_id platform;
    cl_device_id devices[2];
    cl_uint num_devices;

    clGetPlatformIDs(1 , &platform,NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, devices, &num_devices);

    for (int i = 0 ; i < 2 ; i++){
        processors[i].initializeOpenCL(devices[i]);
    }

    all_output_data.resize(all_images_data.size());

    for (int i = 0 ; i < NUM_IMAGES ; i++){
        unsigned char* current_input = all_images_data.data() + (i * single_image_size);
        unsigned char* current_output = all_output_data.data() + (i * single_image_size);

        #pragma omp parallel for num_threads(2)
        for (int gpu = 0 ; gpu < 2; gpu++){
            processors[gpu].processImage(
                current_input,
                current_output,
                width,
                height);
            CImg <unsigned char> output_image(all_output_data.data(), width, height, 1,1);
            output_image.save("output.jpg");
        }
    }
}
