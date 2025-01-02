#include "../include/image_processor.h"

using namespace cimg_library;

ImageProcessor::ImageProcessor(){}

ImageProcessor::loadAndReplicateImage(const char* filename){

    CImg<unsigned char> image(filename);
    width = image.width();
    height = image.height();
}