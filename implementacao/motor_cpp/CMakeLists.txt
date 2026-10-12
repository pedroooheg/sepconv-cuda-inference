cmake_minimum_required(VERSION 3.20)

project(SepConvCpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

# ============================================================
# CUDA TEST
# ============================================================

add_executable(cuda_test
    src/cuda_test.cu
)

# ============================================================
# ONNX RUNTIME
# ============================================================

set(ONNXRUNTIME_ROOT
    "C:/libs/onnxruntime-win-x64-gpu-1.23.2"
)

set(ONNXRUNTIME_INCLUDE_DIR
    "${ONNXRUNTIME_ROOT}/include"
)

set(ONNXRUNTIME_LIB_DIR
    "${ONNXRUNTIME_ROOT}/lib"
)

add_executable(onnx_test
    src/onnx_test.cpp
)

target_include_directories(onnx_test PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_directories(onnx_test PRIVATE
    "${ONNXRUNTIME_LIB_DIR}"
)

target_link_libraries(onnx_test PRIVATE
    onnxruntime
)

# Copia as DLLs do ONNX Runtime para a pasta do executavel.
add_custom_command(TARGET onnx_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:onnx_test>/onnxruntime.dll"

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:onnx_test>/onnxruntime_providers_shared.dll"

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_cuda.dll"
        "$<TARGET_FILE_DIR:onnx_test>/onnxruntime_providers_cuda.dll"
)

# ============================================================
# MOTOR SEPCONV (OpenCV + ONNX Runtime + CUDA)
# ============================================================

find_package(CUDAToolkit REQUIRED)

set(OpenCV_DIR "C:/libs/opencv/build")
find_package(OpenCV REQUIRED)

# DLLs do CUDA 12 / cuDNN 9 exigidas pelo CUDA Execution Provider do
# ONNX Runtime 1.23. Por padrao usa as que acompanham o PyTorch do venv;
# pode ser sobrescrito em tempo de execucao pela variavel SEPCONV_CUDA_DLL_DIR.
set(SEPCONV_CUDA_DLL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../venv/Lib/site-packages/torch/lib"
    CACHE PATH "Diretorio com cudart64_12, cublas64_12 e cudnn64_9"
)

add_executable(sepconv_engine
    src/sepconv_engine.cpp
    src/sepconv_kernels.cu
)

target_include_directories(sepconv_engine PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_directories(sepconv_engine PRIVATE
    "${ONNXRUNTIME_LIB_DIR}"
)

target_link_libraries(sepconv_engine PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::nvml
)

target_compile_definitions(sepconv_engine PRIVATE
    SEPCONV_CUDA_DLL_DIR="${SEPCONV_CUDA_DLL_DIR}"
)

target_compile_options(sepconv_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

add_custom_command(TARGET sepconv_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.dll"
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_cuda.dll"
        "C:/libs/opencv/build/x64/vc16/bin/opencv_world4130.dll"
        "C:/libs/opencv/build/x64/vc16/bin/opencv_videoio_ffmpeg4130_64.dll"
        "$<TARGET_FILE_DIR:sepconv_engine>"
)
