extern "C" __global__ void blur_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // 3x3 box blur: sum neighbour pixels (clamped at edges)
    int r = 0, g = 0, b = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = x + dx;
            int sy = y + dy;
            if (sx < 0 || sx >= width) sx = x;
            if (sy < 0 || sy >= height) sy = y;
            int idx = sy * stride + sx * 4;
            b += (int)data[idx + 0];
            g += (int)data[idx + 1];
            r += (int)data[idx + 2];
            ++count;
        }
    }

    int idx = y * stride + x * 4;
    data[idx + 0] = (unsigned char)(b / count);
    data[idx + 1] = (unsigned char)(g / count);
    data[idx + 2] = (unsigned char)(r / count);
}
