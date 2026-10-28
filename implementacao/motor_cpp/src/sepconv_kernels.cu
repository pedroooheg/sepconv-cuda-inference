#include "sepconv_kernels.cuh"

namespace
{

__global__ void bgr8_to_planar_float_kernel(
    const std::uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int width,
    int height
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const std::size_t plane = static_cast<std::size_t>(width) * height;
    const std::size_t pixel = static_cast<std::size_t>(y) * width + x;

    for (int c = 0; c < 3; ++c)
    {
        dst[c * plane + pixel] = src[pixel * 3 + c] * (1.0f / 255.0f);
    }
}

__global__ void frame_sums_kernel(
    const float* __restrict__ data,
    std::size_t count,
    double* __restrict__ partials
)
{
    __shared__ double shared_sum[256];
    __shared__ double shared_squares[256];

    double sum = 0.0;
    double squares = 0.0;

    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         i < count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    {
        const double value = data[i];
        sum += value;
        squares += value * value;
    }

    shared_sum[threadIdx.x] = sum;
    shared_squares[threadIdx.x] = squares;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_squares[threadIdx.x] += shared_squares[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        partials[2 * blockIdx.x] = shared_sum[0];
        partials[2 * blockIdx.x + 1] = shared_squares[0];
    }
}

// Acumula a contribuicao de um quadro para o pixel (x, y).
//
// Como cada pixel possui o seu proprio par de filtros (v, h), a soma e feita
// linha a linha: para cada linha fy do filtro vertical calcula-se a soma
// ponderada pelo filtro horizontal e depois multiplica-se por v[fy].
// O filtro horizontal fica em registradores durante todo o laco.
//
// Retorna a soma dos pesos do kernel 2D (sum(v) * sum(h)), equivalente ao
// canal de "uns" que o codigo original concatena para normalizar a saida.
template <int K>
__device__ float accumulate_frame(
    const float* __restrict__ frame,
    const float* __restrict__ ver,
    const float* __restrict__ hor,
    std::size_t filter_plane,
    std::size_t filter_pixel,
    int x,
    int y,
    int width,
    int height,
    float acc[3]
)
{
    constexpr int R = K / 2;

    const std::size_t plane = static_cast<std::size_t>(width) * height;

    float h[K];
    int sx[K];
    float hor_sum = 0.0f;

    #pragma unroll
    for (int fx = 0; fx < K; ++fx)
    {
        h[fx] = hor[fx * filter_plane + filter_pixel];
        sx[fx] = min(max(x + fx - R, 0), width - 1);
        hor_sum += h[fx];
    }

    float ver_sum = 0.0f;

    for (int fy = 0; fy < K; ++fy)
    {
        const float v = ver[fy * filter_plane + filter_pixel];
        ver_sum += v;

        const int sy = min(max(y + fy - R, 0), height - 1);
        const float* row = frame + static_cast<std::size_t>(sy) * width;

        float row_b = 0.0f;
        float row_g = 0.0f;
        float row_r = 0.0f;

        #pragma unroll
        for (int fx = 0; fx < K; ++fx)
        {
            row_b += h[fx] * row[sx[fx]];
            row_g += h[fx] * row[plane + sx[fx]];
            row_r += h[fx] * row[2 * plane + sx[fx]];
        }

        acc[0] += v * row_b;
        acc[1] += v * row_g;
        acc[2] += v * row_r;
    }

    return ver_sum * hor_sum;
}

template <int K>
__global__ void __launch_bounds__(256) sepconv_kernel(
    const float* __restrict__ frame1,
    const float* __restrict__ frame2,
    int width,
    int height,
    const float* __restrict__ ver1,
    const float* __restrict__ hor1,
    const float* __restrict__ ver2,
    const float* __restrict__ hor2,
    SepconvRegion region,
    std::uint8_t* __restrict__ out,
    int out_width,
    int out_height
)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = region.x0 + tx;
    const int y = region.y0 + ty;

    if (tx >= region.width || ty >= region.height || x >= out_width || y >= out_height)
    {
        return;
    }

    const std::size_t filter_plane =
        static_cast<std::size_t>(region.filter_width) * region.filter_height;
    const std::size_t filter_pixel =
        static_cast<std::size_t>(region.filter_y0 + ty) * region.filter_width + region.filter_x0 + tx;

    float acc[3] = { 0.0f, 0.0f, 0.0f };

    float norm =
        accumulate_frame<K>(frame1, ver1, hor1, filter_plane, filter_pixel, x, y, width, height, acc) +
        accumulate_frame<K>(frame2, ver2, hor2, filter_plane, filter_pixel, x, y, width, height, acc);

    // Mesmo criterio do run.py: evita divisao por valores proximos de zero.
    if (fabsf(norm) < 0.01f)
    {
        norm = 1.0f;
    }

    const std::size_t out_pixel =
        (static_cast<std::size_t>(y) * out_width + x) * 3;

    for (int c = 0; c < 3; ++c)
    {
        const float value = fminf(fmaxf(acc[c] / norm, 0.0f), 1.0f);

        // Truncamento, como em (x * 255.0).astype(uint8) no run.py.
        out[out_pixel + c] = static_cast<std::uint8_t>(value * 255.0f);
    }
}

dim3 grid_for(int width, int height, dim3 block)
{
    return dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
}

} // namespace

void launch_bgr8_to_planar_float(
    const std::uint8_t* src,
    float* dst,
    int width,
    int height,
    cudaStream_t stream
)
{
    const dim3 block(32, 8);

    bgr8_to_planar_float_kernel<<<grid_for(width, height, block), block, 0, stream>>>(
        src, dst, width, height
    );
}

void frame_sums(
    const float* data,
    std::size_t count,
    double* partials,
    double& sum,
    double& sum_of_squares,
    cudaStream_t stream
)
{
    frame_sums_kernel<<<SEPCONV_REDUCTION_BLOCKS, 256, 0, stream>>>(data, count, partials);

    double host[2 * SEPCONV_REDUCTION_BLOCKS];
    cudaMemcpyAsync(host, partials, sizeof(host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    sum = 0.0;
    sum_of_squares = 0.0;

    for (int i = 0; i < SEPCONV_REDUCTION_BLOCKS; ++i)
    {
        sum += host[2 * i];
        sum_of_squares += host[2 * i + 1];
    }
}

void launch_sepconv(
    const float* frame1,
    const float* frame2,
    int width,
    int height,
    const float* ver1,
    const float* hor1,
    const float* ver2,
    const float* hor2,
    const SepconvRegion& region,
    std::uint8_t* out,
    int out_width,
    int out_height,
    cudaStream_t stream
)
{
    const dim3 block(32, 8);

    sepconv_kernel<SEPCONV_KERNEL_SIZE><<<grid_for(region.width, region.height, block), block, 0, stream>>>(
        frame1, frame2, width, height, ver1, hor1, ver2, hor2,
        region, out, out_width, out_height
    );
}
