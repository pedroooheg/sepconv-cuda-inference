#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_cuda()
{
    printf(
        "Hello from CUDA! Block %d, Thread %d\n",
        blockIdx.x,
        threadIdx.x
    );
}

int main()
{
    int device_count = 0;

    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess)
    {
        std::cerr << "Erro ao detectar GPU CUDA: "
                  << cudaGetErrorString(error)
                  << std::endl;

        return 1;
    }

    if (device_count == 0)
    {
        std::cerr << "Nenhuma GPU CUDA encontrada." << std::endl;
        return 1;
    }

    cudaDeviceProp properties{};

    error = cudaGetDeviceProperties(&properties, 0);

    if (error != cudaSuccess)
    {
        std::cerr << "Erro ao obter propriedades da GPU: "
                  << cudaGetErrorString(error)
                  << std::endl;

        return 1;
    }

    std::cout << "GPU detectada: "
              << properties.name
              << std::endl;

    std::cout << "Compute Capability: "
              << properties.major
              << "."
              << properties.minor
              << std::endl;

    hello_cuda<<<1, 5>>>();

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        std::cerr << "Erro ao executar kernel: "
                  << cudaGetErrorString(error)
                  << std::endl;

        return 1;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        std::cerr << "Erro durante sincronização CUDA: "
                  << cudaGetErrorString(error)
                  << std::endl;

        return 1;
    }

    std::cout << "CUDA executado com sucesso!" << std::endl;

    return 0;
}