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
    int x,
    int y,
    int width,
    int height,
    float acc[3]
)
{
    constexpr int R = K / 2;

    const std::size_t plane = static_cast<std::size_t>(width) * height;
    const std::size_t pixel = static_cast<std::size_t>(y) * width + x;

    float h[K];
    int sx[K];
    float hor_sum = 0.0f;

    #pragma unroll
    for (int fx = 0; fx < K; ++fx)
    {
        h[fx] = hor[fx * plane + pixel];
        sx[fx] = min(max(x + fx - R, 0), width - 1);
        hor_sum += h[fx];
    }

    float ver_sum = 0.0f;

    for (int fy = 0; fy < K; ++fy)
    {
        const float v = ver[fy * plane + pixel];
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
    const float* __restrict__ ver1,
    const float* __restrict__ hor1,
    const float* __restrict__ ver2,
    const float* __restrict__ hor2,
    std::uint8_t* __restrict__ out,
    int out_width,
    int out_height,
    int width,
    int height
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height)
    {
        return;
    }

    float acc[3] = { 0.0f, 0.0f, 0.0f };

    float norm =
        accumulate_frame<K>(frame1, ver1, hor1, x, y, width, height, acc) +
        accumulate_frame<K>(frame2, ver2, hor2, x, y, width, height, acc);

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

void launch_sepconv(
    const float* frame1,
    const float* frame2,
    const float* ver1,
    const float* hor1,
    const float* ver2,
    const float* hor2,
    std::uint8_t* out,
    int out_width,
    int out_height,
    int width,
    int height,
    cudaStream_t stream
)
{
    const dim3 block(32, 8);

    sepconv_kernel<SEPCONV_KERNEL_SIZE><<<grid_for(out_width, out_height, block), block, 0, stream>>>(
        frame1, frame2, ver1, hor1, ver2, hor2,
        out, out_width, out_height, width, height
    );
}
