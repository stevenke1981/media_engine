extern "C" __global__ void threshold_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * stride + x * 4;
    int mean = ((int)data[idx + 0] + (int)data[idx + 1] + (int)data[idx + 2]) / 3;
    unsigned char val = (mean > (int)(threshold * 255.0f)) ? (unsigned char)255 : (unsigned char)0;

    data[idx + 0] = val;
    data[idx + 1] = val;
    data[idx + 2] = val;
}
