extern "C" __global__ void box_blur_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride,
    float radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int r = (int)(radius + 0.5f);
    if (r < 1) r = 1;

    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int sx = x + dx;
            int sy = y + dy;
            if (sx < 0) sx = 0;
            if (sx >= width) sx = width - 1;
            if (sy < 0) sy = 0;
            if (sy >= height) sy = height - 1;

            int sidx = sy * stride + sx * 4;
            sum_r += (int)data[sidx + 0];
            sum_g += (int)data[sidx + 1];
            sum_b += (int)data[sidx + 2];
            count++;
        }
    }

    int idx = y * stride + x * 4;
    data[idx + 0] = (unsigned char)(sum_r / count);
    data[idx + 1] = (unsigned char)(sum_g / count);
    data[idx + 2] = (unsigned char)(sum_b / count);
}
