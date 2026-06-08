extern "C" __global__ void sharpen_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride,
    float strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Compute 3x3 box blur average for the center pixel
    int r_sum = 0, g_sum = 0, b_sum = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = x + dx;
            int sy = y + dy;
            if (sx < 0 || sx >= width) sx = x;
            if (sy < 0 || sy >= height) sy = y;
            int idx = sy * stride + sx * 4;
            b_sum += (int)data[idx + 0];
            g_sum += (int)data[idx + 1];
            r_sum += (int)data[idx + 2];
            ++count;
        }
    }

    int idx = y * stride + x * 4;
    int center_b = (int)data[idx + 0];
    int center_g = (int)data[idx + 1];
    int center_r = (int)data[idx + 2];

    int avg_b = b_sum / count;
    int avg_g = g_sum / count;
    int avg_r = r_sum / count;

    // Unsharp mask: out = center + (center - blur) * strength
    int out_b = (int)((float)center_b + ((float)center_b - (float)avg_b) * strength);
    int out_g = (int)((float)center_g + ((float)center_g - (float)avg_g) * strength);
    int out_r = (int)((float)center_r + ((float)center_r - (float)avg_r) * strength);

    data[idx + 0] = (unsigned char)fminf(fmaxf((float)out_b, 0.0f), 255.0f);
    data[idx + 1] = (unsigned char)fminf(fmaxf((float)out_g, 0.0f), 255.0f);
    data[idx + 2] = (unsigned char)fminf(fmaxf((float)out_r, 0.0f), 255.0f);
}
