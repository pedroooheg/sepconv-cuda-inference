#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Tamanho dos filtros 1D estimados pela rede SepConv (51 coeficientes).
constexpr int SEPCONV_KERNEL_SIZE = 51;

// Converte um quadro BGR intercalado (uint8, HWC) em float planar (CHW)
// no intervalo [0, 1], formato esperado pela rede.
void launch_bgr8_to_planar_float(
    const std::uint8_t* src,
    float* dst,
    int width,
    int height,
    cudaStream_t stream
);

// Soma e soma dos quadrados de `count` valores, usadas para calcular a media
// e o desvio padrao do quadro inteiro (normalizacao do processamento em
// blocos). `partials` e um buffer de trabalho na VRAM com
// 2 * SEPCONV_REDUCTION_BLOCKS doubles. Sincroniza o stream.
constexpr int SEPCONV_REDUCTION_BLOCKS = 256;

void frame_sums(
    const float* data,
    std::size_t count,
    double* partials,
    double& sum,
    double& sum_of_squares,
    cudaStream_t stream
);

// Regiao do quadro de saida calculada por uma chamada de launch_sepconv e
// posicao correspondente dentro dos tensores de filtros.
//
// No quadro inteiro, a regiao e o quadro todo e os filtros tem o tamanho do
// quadro. No processamento em blocos, a regiao e o miolo do bloco e os filtros
// tem o tamanho do bloco com margem.
struct SepconvRegion
{
    int x0;
    int y0;
    int width;
    int height;
    int filter_x0;
    int filter_y0;
    int filter_width;
    int filter_height;
};

// Aplica a convolucao adaptativa separavel nos dois quadros de entrada e
// escreve a regiao do quadro intermediario ja convertida para BGR uint8.
//
// frame1/frame2: [3, height, width]                        (float planar)
// ver*/hor*:     [51, filter_height, filter_width]          (saidas do ONNX)
// out:           [out_height, out_width, 3]                 (uint8 intercalado)
//
// A borda "replicate" de 25 pixels do original e obtida limitando as
// coordenadas de leitura, sem alocar copias dos quadros com padding na VRAM.
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
);
