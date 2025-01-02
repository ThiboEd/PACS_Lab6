#include "../include/image_processor.h"

int main(){

    ImageProcessor img_process;

    img_process.loadAndReplicateImage("image/image.jpg");

    img_process.processImagesWithOpenCL();

    return EXIT_SUCCESS;
}
