extern "C" __global__ void pixelate_kernel(
    unsigned char* data,
    int width,
    int height,
    int stride,
    float block_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int bs = (int)(block_size + 0.5f);
    if (bs < 1) bs = 1;

    // Map to top-left of the block
    int bx = (x / bs) * bs;
    int by = (y / bs) * bs;

    int src_idx = by * stride + bx * 4;
    int dst_idx = y * stride + x * 4;

    data[dst_idx + 0] = data[src_idx + 0];
    data[dst_idx + 1] = data[src_idx + 1];
    data[dst_idx + 2] = data[src_idx + 2];
}
