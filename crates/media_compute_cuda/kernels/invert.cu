extern "C" __global__ void invert_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * stride + x * 4;
    data[idx + 0] = 255 - data[idx + 0];
    data[idx + 1] = 255 - data[idx + 1];
    data[idx + 2] = 255 - data[idx + 2];
}
