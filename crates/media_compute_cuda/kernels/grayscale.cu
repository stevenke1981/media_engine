extern "C" __global__ void grayscale_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * stride + x * 4;
    float gray = (float)data[idx + 0] * 0.2126f
               + (float)data[idx + 1] * 0.7152f
               + (float)data[idx + 2] * 0.0722f;
    unsigned char g = (unsigned char)fminf(fmaxf(gray, 0.0f), 255.0f);
    data[idx + 0] = g;
    data[idx + 1] = g;
    data[idx + 2] = g;
}
