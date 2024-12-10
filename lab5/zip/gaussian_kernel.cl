bool check_borders (int x , int y , int width , int height){
    /*
    Return true if pixels belongs to border, false otherwise
    */
    return (x == 0 || y == 0 || x == width - 1 || y == height - 1);
}

__kernel void gaussian_blur(global uchar* image,
                            global uchar* output_image,
                            global float* gaussian_kernel,
                            global int* image_dim){
                                
    const int height = image_dim[0];
    const int width = image_dim[1];

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (check_borders(x , y , width , height)){
        output_image[y * width + x] = image[y * width + x];
        return;
    }

    float sum = 0.0f;

    for (int i = -1 ; i <=1 ; i++){
        for (int j = -1 ; j <= 1 ; j++){
            int nx = x + i;
            int ny = y + j;
            sum += gaussian_kernel[(i+1)*3 + (j+1)] * image[ny * width +nx];
        }
    output_image[y * width + x] = (uchar)sum;
    }
}
