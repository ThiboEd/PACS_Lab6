#include "../include/image_processor.h"

int main() {
    ImageProcessor img_process;

    img_process.loadAndReplicateImage("image/image.jpg");

    GlobalMetrics metrics = img_process.processImagesWithOpenCL();
    img_process.printMetrics(metrics);

    return EXIT_SUCCESS;
}

