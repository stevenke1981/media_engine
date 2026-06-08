extern "C" __global__ void edge_detect_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int gx_s = 0, gy_s = 0;

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int nx = x + kx - 1;
            int ny = y + ky - 1;
            if (nx < 0) nx = 0;
            if (nx >= width) nx = width - 1;
            if (ny < 0) ny = 0;
            if (ny >= height) ny = height - 1;

            int nidx = ny * stride + nx * 4;
            float luma = (float)data[nidx + 0] * 0.2126f
                       + (float)data[nidx + 1] * 0.7152f
                       + (float)data[nidx + 2] * 0.0722f;
            int pval = (int)luma;

            // Sobel X kernel: [-1,0,1; -2,0,2; -1,0,1]
            // Sobel Y kernel: [-1,-2,-1; 0,0,0; 1,2,1]
            int sx = (kx == 0 ? -1 : (kx == 2 ? 1 : 0));
            int sy = (ky == 0 ? -1 : (ky == 2 ? 1 : 0));
            gx_s += pval * sx * (ky == 1 ? 2 : 1);
            gy_s += pval * sy * (kx == 1 ? 2 : 1);
        }
    }

    float mag = fminf(fmaxf(sqrtf((float)(gx_s * gx_s + gy_s * gy_s)), 0.0f), 255.0f);
    unsigned char c = (unsigned char)mag;

    int idx = y * stride + x * 4;
    data[idx + 0] = c;
    data[idx + 1] = c;
    data[idx + 2] = c;
}
