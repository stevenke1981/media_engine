extern "C" __global__ void brightness_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride,
    float factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * stride + x * 4;
    for (int c = 0; c < 3; ++c) {
        float v = (float)data[idx + c] + factor * 255.0f;
        data[idx + c] = (unsigned char)fminf(fmaxf(v, 0.0f), 255.0f);
    }
}
