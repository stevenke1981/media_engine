extern "C" __global__ void sepia_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * stride + x * 4;
    float r = (float)data[idx + 0];
    float g = (float)data[idx + 1];
    float b = (float)data[idx + 2];

    float out_r = r * 0.393f + g * 0.769f + b * 0.189f;
    float out_g = r * 0.349f + g * 0.686f + b * 0.168f;
    float out_b = r * 0.272f + g * 0.534f + b * 0.131f;

    data[idx + 0] = (unsigned char)fminf(fmaxf(out_r, 0.0f), 255.0f);
    data[idx + 1] = (unsigned char)fminf(fmaxf(out_g, 0.0f), 255.0f);
    data[idx + 2] = (unsigned char)fminf(fmaxf(out_b, 0.0f), 255.0f);
}
