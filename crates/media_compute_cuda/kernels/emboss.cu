extern "C" __global__ void emboss_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int acc = 0;

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

            // Emboss kernel: [-2,-1,0; -1,1,1; 0,1,2]
            static const int emb_k[3][3] = {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};
            acc += (int)luma * emb_k[ky][kx];
        }
    }

    int val = acc + 128;
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    unsigned char c = (unsigned char)val;

    int idx = y * stride + x * 4;
    data[idx + 0] = c;
    data[idx + 1] = c;
    data[idx + 2] = c;
}
