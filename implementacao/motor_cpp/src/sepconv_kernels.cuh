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

// Aplica a convolucao adaptativa separavel nos dois quadros de entrada e
// escreve o quadro intermediario ja convertido para BGR uint8.
//
// frame1/frame2: [3, height, width]   (float planar)
// ver*/hor*:     [51, height, width]  (saidas do ONNX)
// out:           [out_height, out_width, 3] (uint8 intercalado)
//
// A borda "replicate" de 25 pixels do original e obtida limitando as
// coordenadas de leitura, sem alocar copias dos quadros com padding na VRAM.
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
);
