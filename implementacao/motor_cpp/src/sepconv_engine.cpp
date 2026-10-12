// Motor de inferencia SepConv nativo em C++/CUDA.
//
// Pipeline de cada quadro interpolado:
//   1. I/O (Host)          OpenCV le os quadros (BGR uint8)
//   2. Host -> Device      copia dos quadros uint8 para a VRAM
//   3. Pre-processamento   kernel CUDA converte para float planar [0, 1]
//   4. Inferencia          ONNX Runtime (CUDA EP) gera os filtros 1D na VRAM
//   5. Convolucao          kernel CUDA aplica a convolucao separavel
//   6. Device -> Host      copia apenas o quadro final (uint8) e grava
//
// Modo imagem (um par de quadros, repetido --runs vezes):
//   sepconv_engine.exe --model m.onnx --one a.png --two b.png --out out.png
//                      [--runs 10] [--warmup 3]
//
// Modo video (interpola todos os pares consecutivos):
//   sepconv_engine.exe --model m.onnx --video in.mkv --out out.mkv
//                      [--factor 2|4] [--max-frames N] [--warmup 3]
//
// Opcoes comuns:
//   --csv arquivo.csv          acrescenta uma linha de resumo ao CSV
//   --ort-option chave=valor   repassa opcoes ao CUDA Execution Provider,
//                              ex.: cudnn_conv_algo_search=HEURISTIC

#include "sepconv_kernels.cuh"

#include <cuda_runtime.h>
#include <nvml.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void check_cuda(cudaError_t error, const char* what)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(error)
        );
    }
}

struct Arguments
{
    std::string model;
    std::string one;
    std::string two;
    std::string video;
    std::string out;
    std::string csv;
    int runs = 10;
    int warmup = 3;
    int factor = 2;
    int max_frames = 0;
    std::vector<std::pair<std::string, std::string>> ort_options;
};

Arguments parse_arguments(int argc, char* argv[])
{
    Arguments args;

    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string key = argv[i];
        const std::string value = argv[i + 1];

        if (key == "--model") args.model = value;
        else if (key == "--one") args.one = value;
        else if (key == "--two") args.two = value;
        else if (key == "--video") args.video = value;
        else if (key == "--out") args.out = value;
        else if (key == "--csv") args.csv = value;
        else if (key == "--runs") args.runs = std::stoi(value);
        else if (key == "--warmup") args.warmup = std::stoi(value);
        else if (key == "--factor") args.factor = std::stoi(value);
        else if (key == "--max-frames") args.max_frames = std::stoi(value);
        else if (key == "--ort-option")
        {
            const std::size_t separator = value.find('=');

            if (separator == std::string::npos)
            {
                throw std::runtime_error("--ort-option espera chave=valor: " + value);
            }

            args.ort_options.emplace_back(value.substr(0, separator), value.substr(separator + 1));
        }
        else throw std::runtime_error("argumento desconhecido: " + key);
    }

    const bool image_mode = !args.one.empty() && !args.two.empty();
    const bool video_mode = !args.video.empty();

    if (args.model.empty() || args.out.empty() || image_mode == video_mode)
    {
        throw std::runtime_error(
            "uso:\n"
            "  sepconv_engine.exe --model m.onnx --one a.png --two b.png --out out.png [--runs N] [--warmup N]\n"
            "  sepconv_engine.exe --model m.onnx --video in.mkv --out out.mkv [--factor 2|4] [--max-frames N] [--warmup N]\n"
            "  opcoes comuns: [--csv resumo.csv] [--ort-option chave=valor]"
        );
    }

    if (args.factor != 2 && args.factor != 4)
    {
        throw std::runtime_error("--factor deve ser 2 ou 4");
    }

    return args;
}

// O ONNX Runtime 1.23 (GPU) depende das DLLs do CUDA 12 e do cuDNN 9.
// Elas sao procuradas no PATH quando o CUDA Execution Provider e carregado,
// entao o diretorio configurado e adicionado ao PATH do proprio processo.
void add_cuda_dll_directory()
{
    std::wstring directory;

    if (const wchar_t* from_env = _wgetenv(L"SEPCONV_CUDA_DLL_DIR"))
    {
        directory = from_env;
    }
#ifdef SEPCONV_CUDA_DLL_DIR
    else
    {
        directory = std::filesystem::path(SEPCONV_CUDA_DLL_DIR).wstring();
    }
#endif

    if (directory.empty() || !std::filesystem::exists(directory))
    {
        return;
    }

    const wchar_t* current = _wgetenv(L"PATH");
    std::wstring path = directory + L";" + (current ? current : L"");

    SetEnvironmentVariableW(L"PATH", path.c_str());
}

// Memoria de video da GPU inteira (MiB), segundo a NVML.
// No Windows (WDDM) a NVML nao informa o uso por processo, por isso o
// consumo do motor e obtido pela diferenca em relacao a linha de base.
struct VramMonitor
{
    nvmlDevice_t device{};
    double total_mib = 0.0;
    double baseline_mib = 0.0;
    double peak_mib = 0.0;

    VramMonitor()
    {
        if (nvmlInit_v2() != NVML_SUCCESS)
        {
            throw std::runtime_error("falha ao inicializar a NVML");
        }

        nvmlDeviceGetHandleByIndex_v2(0, &device);

        nvmlMemory_t memory{};
        nvmlDeviceGetMemoryInfo(device, &memory);

        total_mib = memory.total / (1024.0 * 1024.0);
        baseline_mib = memory.used / (1024.0 * 1024.0);
        peak_mib = baseline_mib;
    }

    ~VramMonitor()
    {
        nvmlShutdown();
    }

    void sample()
    {
        nvmlMemory_t memory{};
        nvmlDeviceGetMemoryInfo(device, &memory);
        peak_mib = std::max(peak_mib, memory.used / (1024.0 * 1024.0));
    }

    // Com a VRAM praticamente cheia o driver do Windows passa a usar a
    // memoria compartilhada do sistema (Sysmem Fallback), o que aumenta
    // muito a latencia. Esses resultados precisam ser sinalizados.
    bool saturated() const
    {
        return peak_mib >= 0.97 * total_mib;
    }
};

struct StageTimes
{
    std::vector<double> upload;
    std::vector<double> preprocess;
    std::vector<double> inference;
    std::vector<double> sepconv;
    std::vector<double> download;
    std::vector<double> total;
};

double mean(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }

    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double stddev(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const double m = mean(values);
    double sum = 0.0;

    for (double v : values)
    {
        sum += (v - m) * (v - m);
    }

    return std::sqrt(sum / values.size());
}

void print_stage(const char* name, const std::vector<double>& values)
{
    std::cout << "  " << std::left << std::setw(20) << name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << mean(values) << " ms  (+/- "
              << stddev(values) << ")\n";
}

// =============================================================
// Motor: aloca todos os buffers uma unica vez para a resolucao do
// video e reutiliza-os em todas as interpolacoes.
// =============================================================

class SepConvEngine
{
public:
    SepConvEngine(
        const Arguments& args,
        int out_width,
        int out_height,
        VramMonitor& vram
    )
        : out_width_(out_width),
          out_height_(out_height),
          // A rede exige dimensoes pares (mesmo padding replicate do run.py).
          width_(out_width + (out_width % 2)),
          height_(out_height + (out_height % 2)),
          vram_(vram),
          env_(ORT_LOGGING_LEVEL_ERROR, "SepConvEngine"),
          cuda_memory_("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
    {
        allocate_buffers();
        create_session(args);
        bind_tensors();

        for (cudaEvent_t& event : events_)
        {
            check_cuda(cudaEventCreate(&event), "cudaEventCreate");
        }
    }

    ~SepConvEngine()
    {
        for (cudaEvent_t event : events_)
        {
            cudaEventDestroy(event);
        }

        cudaFree(d_bgr_one_);
        cudaFree(d_bgr_two_);
        cudaFree(d_frame_one_);
        cudaFree(d_frame_two_);

        for (float* filter : d_filters_)
        {
            cudaFree(filter);
        }

        cudaFree(d_out_);
    }

    SepConvEngine(const SepConvEngine&) = delete;
    SepConvEngine& operator=(const SepConvEngine&) = delete;

    int width() const { return width_; }
    int height() const { return height_; }

    double buffers_mib() const
    {
        const double pixels = static_cast<double>(width_) * height_;

        return (2.0 * pixels * 3 +                                   // quadros uint8
                2.0 * pixels * 3 * sizeof(float) +                   // quadros float
                4.0 * pixels * SEPCONV_KERNEL_SIZE * sizeof(float) + // filtros 1D
                static_cast<double>(out_width_) * out_height_ * 3)   // saida uint8
               / (1024.0 * 1024.0);
    }

    // Sintetiza o quadro intermediario entre one e two.
    // Se times != nullptr, registra o tempo de cada estagio.
    void interpolate(const cv::Mat& one, const cv::Mat& two, cv::Mat& result, StageTimes* times)
    {
        const cv::Mat& padded_one = pad_to_network(one, padded_one_);
        const cv::Mat& padded_two = pad_to_network(two, padded_two_);

        result.create(out_height_, out_width_, CV_8UC3);

        const std::size_t bgr_bytes = static_cast<std::size_t>(width_) * height_ * 3;
        const std::size_t out_bytes = static_cast<std::size_t>(out_width_) * out_height_ * 3;

        const auto start = Clock::now();

        // 2. Host -> Device
        cudaEventRecord(events_[0]);
        check_cuda(cudaMemcpy(d_bgr_one_, padded_one.data, bgr_bytes, cudaMemcpyHostToDevice), "upload one");
        check_cuda(cudaMemcpy(d_bgr_two_, padded_two.data, bgr_bytes, cudaMemcpyHostToDevice), "upload two");

        // 3. Pre-processamento
        cudaEventRecord(events_[1]);
        launch_bgr8_to_planar_float(d_bgr_one_, d_frame_one_, width_, height_, nullptr);
        launch_bgr8_to_planar_float(d_bgr_two_, d_frame_two_, width_, height_, nullptr);
        cudaEventRecord(events_[2]);
        check_cuda(cudaDeviceSynchronize(), "pre-processamento");

        // 4. Inferencia (o ONNX Runtime usa o proprio stream e so retorna
        //    depois que a GPU termina, por isso o tempo e medido no Host)
        const auto inference_start = Clock::now();
        session_.Run(run_options_, binding_);
        const auto inference_end = Clock::now();

        vram_.sample();

        // 5. Convolucao separavel
        // Ordem de saida do ONNX: kernel_v1, kernel_v2, kernel_h1, kernel_h2.
        cudaEventRecord(events_[3]);
        launch_sepconv(d_frame_one_, d_frame_two_,
                       d_filters_[0], d_filters_[2], d_filters_[1], d_filters_[3],
                       d_out_, out_width_, out_height_, width_, height_, nullptr);
        check_cuda(cudaGetLastError(), "launch sepconv");
        cudaEventRecord(events_[4]);

        // 6. Device -> Host
        check_cuda(cudaMemcpy(result.data, d_out_, out_bytes, cudaMemcpyDeviceToHost), "download");

        const auto end = Clock::now();

        if (times == nullptr)
        {
            return;
        }

        float upload_ms = 0.0f;
        float preprocess_ms = 0.0f;
        float sepconv_ms = 0.0f;

        cudaEventElapsedTime(&upload_ms, events_[0], events_[1]);
        cudaEventElapsedTime(&preprocess_ms, events_[1], events_[2]);
        cudaEventElapsedTime(&sepconv_ms, events_[3], events_[4]);

        const double inference_ms = elapsed_ms(inference_start, inference_end);
        const double total_ms = elapsed_ms(start, end);

        times->upload.push_back(upload_ms);
        times->preprocess.push_back(preprocess_ms);
        times->inference.push_back(inference_ms);
        times->sepconv.push_back(sepconv_ms);
        times->total.push_back(total_ms);
        times->download.push_back(
            total_ms - upload_ms - preprocess_ms - inference_ms - sepconv_ms
        );
    }

private:
    const cv::Mat& pad_to_network(const cv::Mat& frame, cv::Mat& buffer) const
    {
        if (frame.cols != out_width_ || frame.rows != out_height_ || frame.type() != CV_8UC3)
        {
            throw std::runtime_error("quadro com resolucao ou formato inesperado");
        }

        if (width_ == out_width_ && height_ == out_height_ && frame.isContinuous())
        {
            return frame;
        }

        cv::copyMakeBorder(frame, buffer, 0, height_ - out_height_, 0, width_ - out_width_, cv::BORDER_REPLICATE);
        return buffer;
    }

    void allocate_buffers()
    {
        const std::size_t pixels = static_cast<std::size_t>(width_) * height_;

        check_cuda(cudaMalloc(&d_bgr_one_, pixels * 3), "cudaMalloc bgr_one");
        check_cuda(cudaMalloc(&d_bgr_two_, pixels * 3), "cudaMalloc bgr_two");
        check_cuda(cudaMalloc(&d_frame_one_, pixels * 3 * sizeof(float)), "cudaMalloc frame_one");
        check_cuda(cudaMalloc(&d_frame_two_, pixels * 3 * sizeof(float)), "cudaMalloc frame_two");

        for (float*& filter : d_filters_)
        {
            check_cuda(cudaMalloc(&filter, pixels * SEPCONV_KERNEL_SIZE * sizeof(float)), "cudaMalloc filtro");
        }

        check_cuda(cudaMalloc(&d_out_, static_cast<std::size_t>(out_width_) * out_height_ * 3), "cudaMalloc out");
    }

    void create_session(const Arguments& args)
    {
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const OrtApi& api = Ort::GetApi();
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));

        std::vector<const char*> option_keys = { "device_id", "arena_extend_strategy" };
        std::vector<const char*> option_values = { "0", "kSameAsRequested" };

        for (const auto& [key, value] : args.ort_options)
        {
            option_keys.push_back(key.c_str());
            option_values.push_back(value.c_str());
            std::cout << "Opcao do CUDA EP: " << key << "=" << value << "\n";
        }

        Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
            cuda_options, option_keys.data(), option_values.data(), option_keys.size()
        ));
        session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
        api.ReleaseCUDAProviderOptions(cuda_options);

        const std::wstring model_path = std::filesystem::path(args.model).wstring();
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
    }

    // I/O binding: entradas e saidas do ONNX apontam diretamente para os
    // buffers alocados pelo motor, sem copias intermediarias pela CPU.
    void bind_tensors()
    {
        const std::size_t pixels = static_cast<std::size_t>(width_) * height_;
        const std::int64_t frame_shape[] = { 1, 3, height_, width_ };
        const std::int64_t filter_shape[] = { 1, SEPCONV_KERNEL_SIZE, height_, width_ };

        binding_ = Ort::IoBinding(session_);

        tensors_.push_back(Ort::Value::CreateTensor<float>(cuda_memory_, d_frame_one_, pixels * 3, frame_shape, 4));
        binding_.BindInput("input_frame1", tensors_.back());

        tensors_.push_back(Ort::Value::CreateTensor<float>(cuda_memory_, d_frame_two_, pixels * 3, frame_shape, 4));
        binding_.BindInput("input_frame2", tensors_.back());

        const char* output_names[4] = { "kernel_v1", "kernel_v2", "kernel_h1", "kernel_h2" };

        for (int i = 0; i < 4; ++i)
        {
            tensors_.push_back(Ort::Value::CreateTensor<float>(
                cuda_memory_, d_filters_[i], pixels * SEPCONV_KERNEL_SIZE, filter_shape, 4
            ));
            binding_.BindOutput(output_names[i], tensors_.back());
        }
    }

    const int out_width_;
    const int out_height_;
    const int width_;
    const int height_;

    VramMonitor& vram_;

    std::uint8_t* d_bgr_one_ = nullptr;
    std::uint8_t* d_bgr_two_ = nullptr;
    float* d_frame_one_ = nullptr;
    float* d_frame_two_ = nullptr;
    float* d_filters_[4] = {};
    std::uint8_t* d_out_ = nullptr;

    cv::Mat padded_one_;
    cv::Mat padded_two_;

    Ort::Env env_;
    Ort::MemoryInfo cuda_memory_;
    Ort::Session session_{ nullptr };
    Ort::IoBinding binding_{ nullptr };
    Ort::RunOptions run_options_;
    std::vector<Ort::Value> tensors_;

    cudaEvent_t events_[5] = {};
};

// =============================================================
// Relatorio e CSV
// =============================================================

struct Summary
{
    std::string mode;
    std::string input;
    int width = 0;
    int height = 0;
    int factor = 0;
    int written_frames = 0;
    double wall_seconds = 0.0;
};

void print_report(const StageTimes& times, const VramMonitor& vram, const SepConvEngine& engine, const Summary& summary)
{
    std::cout << "\nTempo por quadro interpolado (" << times.total.size() << " medicoes):\n";

    print_stage("Host -> Device", times.upload);
    print_stage("Pre-processamento", times.preprocess);
    print_stage("Inferencia ONNX", times.inference);
    print_stage("Convolucao sepconv", times.sepconv);
    print_stage("Device -> Host*", times.download);
    print_stage("TOTAL", times.total);

    std::cout << "  (*) inclui sincronizacoes e overhead de Host\n";
    std::cout << "  Vazao da interpolacao: " << std::setprecision(2)
              << 1000.0 / mean(times.total) << " quadros/s\n";

    if (summary.mode == "video")
    {
        std::cout << "  Tempo total do video (com decodificacao e gravacao): "
                  << summary.wall_seconds << " s  ->  "
                  << summary.written_frames / summary.wall_seconds
                  << " quadros de saida/s\n";
    }

    std::cout << "\nVRAM (NVML, GPU inteira):\n"
              << std::setprecision(1)
              << "  Capacidade:               " << vram.total_mib << " MiB\n"
              << "  Linha de base:            " << vram.baseline_mib << " MiB\n"
              << "  Buffers do motor:         " << engine.buffers_mib() << " MiB (calculado)\n"
              << "  Pico observado:           " << vram.peak_mib << " MiB\n"
              << "  Consumo do motor (pico):  " << vram.peak_mib - vram.baseline_mib << " MiB\n";

    if (vram.saturated())
    {
        std::cout << "\n  ATENCAO: VRAM saturada. O driver pode ter recorrido a memoria\n"
                     "  compartilhada do sistema; os tempos medidos ficam inflados.\n";
    }
}

void append_csv(const std::string& path, const StageTimes& times, const VramMonitor& vram, const Summary& summary)
{
    const bool write_header = !std::filesystem::exists(path);

    if (const auto parent = std::filesystem::path(path).parent_path(); !parent.empty())
    {
        std::filesystem::create_directories(parent);
    }

    std::ofstream csv(path, std::ios::app);

    if (write_header)
    {
        csv << "implementacao,modo,entrada,largura,altura,fator,medicoes,"
               "upload_ms,pre_ms,inferencia_ms,sepconv_ms,download_ms,total_ms,total_desvio_ms,"
               "quadros_saida,tempo_video_s,vram_total_mib,vram_base_mib,vram_pico_mib,vram_consumo_mib,vram_saturada\n";
    }

    csv << std::fixed << std::setprecision(3)
        << "cpp," << summary.mode << ',' << std::filesystem::path(summary.input).filename().string() << ','
        << summary.width << ',' << summary.height << ',' << summary.factor << ',' << times.total.size() << ','
        << mean(times.upload) << ',' << mean(times.preprocess) << ',' << mean(times.inference) << ','
        << mean(times.sepconv) << ',' << mean(times.download) << ',' << mean(times.total) << ','
        << stddev(times.total) << ',' << summary.written_frames << ',' << summary.wall_seconds << ','
        << vram.total_mib << ',' << vram.baseline_mib << ',' << vram.peak_mib << ','
        << vram.peak_mib - vram.baseline_mib << ',' << (vram.saturated() ? 1 : 0) << '\n';

    std::cout << "Resumo acrescentado em: " << path << "\n";
}

// =============================================================
// Modos de execucao
// =============================================================

int run_image_mode(const Arguments& args, VramMonitor& vram)
{
    const cv::Mat one = cv::imread(args.one, cv::IMREAD_COLOR);
    const cv::Mat two = cv::imread(args.two, cv::IMREAD_COLOR);

    if (one.empty() || two.empty())
    {
        throw std::runtime_error("nao foi possivel abrir os quadros de entrada");
    }

    if (one.size() != two.size())
    {
        throw std::runtime_error("os quadros possuem resolucoes diferentes");
    }

    SepConvEngine engine(args, one.cols, one.rows, vram);

    std::cout << "Resolucao: " << one.cols << "x" << one.rows
              << " (rede: " << engine.width() << "x" << engine.height() << ")\n";

    cv::Mat result;
    StageTimes times;

    for (int iteration = 0; iteration < args.warmup + args.runs; ++iteration)
    {
        engine.interpolate(one, two, result, iteration < args.warmup ? nullptr : &times);
    }

    if (!cv::imwrite(args.out, result))
    {
        throw std::runtime_error("nao foi possivel gravar " + args.out);
    }

    Summary summary{ "imagem", args.one, one.cols, one.rows, 2, 1, 0.0 };
    print_report(times, vram, engine, summary);

    if (!args.csv.empty())
    {
        append_csv(args.csv, times, vram, summary);
    }

    std::cout << "\nQuadro salvo em: " << args.out << "\n";
    return 0;
}

// Codec sem perdas (FFV1) para .mkv/.avi, para que a avaliacao de
// qualidade nao seja contaminada pela compressao do video de saida.
int fourcc_for(const std::string& path)
{
    const std::string extension = std::filesystem::path(path).extension().string();

    if (extension == ".mkv" || extension == ".avi")
    {
        return cv::VideoWriter::fourcc('F', 'F', 'V', '1');
    }

    return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
}

int run_video_mode(const Arguments& args, VramMonitor& vram)
{
    cv::VideoCapture capture(args.video);

    if (!capture.isOpened())
    {
        throw std::runtime_error("nao foi possivel abrir o video " + args.video);
    }

    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = capture.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    if (args.max_frames > 0)
    {
        total_frames = std::min(total_frames, args.max_frames);
    }

    std::cout << "Video: " << width << "x" << height << " @ " << fps << " FPS, "
              << total_frames << " quadros de entrada, fator " << args.factor << "x\n";

    SepConvEngine engine(args, width, height, vram);

    const auto start = Clock::now();

    cv::VideoWriter writer(args.out, fourcc_for(args.out), fps * args.factor, cv::Size(width, height));

    if (!writer.isOpened())
    {
        throw std::runtime_error("nao foi possivel criar o video " + args.out);
    }

    cv::Mat previous;
    cv::Mat next;
    cv::Mat middle;
    cv::Mat quarter;
    cv::Mat three_quarters;

    StageTimes times;
    int interpolations = 0;
    int written = 0;

    // Os primeiros quadros sintetizados servem de aquecimento e nao entram
    // nas estatisticas.
    auto measure = [&]() -> StageTimes*
    {
        return interpolations++ < args.warmup ? nullptr : &times;
    };

    capture >> previous;

    for (int index = 1; index < total_frames; ++index)
    {
        capture >> next;

        if (next.empty())
        {
            break;
        }

        writer << previous;
        ++written;

        // Mesmo esquema do run.py: o quadro t+0.5 e sintetizado a partir
        // dos originais e, no fator 4, t+0.25 e t+0.75 a partir dele.
        engine.interpolate(previous, next, middle, measure());

        if (args.factor == 4)
        {
            engine.interpolate(previous, middle, quarter, measure());
            engine.interpolate(middle, next, three_quarters, measure());

            writer << quarter << middle << three_quarters;
            written += 3;
        }
        else
        {
            writer << middle;
            ++written;
        }

        std::swap(previous, next);

        std::cout << "\r  par " << index << "/" << total_frames - 1 << std::flush;
    }

    writer << previous;
    ++written;
    writer.release();

    const double wall_seconds = elapsed_ms(start, Clock::now()) / 1000.0;

    std::cout << "\n";

    Summary summary{ "video", args.video, width, height, args.factor, written, wall_seconds };
    print_report(times, vram, engine, summary);

    if (!args.csv.empty())
    {
        append_csv(args.csv, times, vram, summary);
    }

    std::cout << "\nVideo salvo em: " << args.out << " (" << written << " quadros, "
              << fps * args.factor << " FPS)\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const Arguments args = parse_arguments(argc, argv);

        add_cuda_dll_directory();

        // Linha de base de memoria antes de qualquer alocacao do motor.
        VramMonitor vram;

        return args.video.empty()
            ? run_image_mode(args, vram)
            : run_video_mode(args, vram);
    }
    catch (const Ort::Exception& exception)
    {
        std::cerr << "\nERRO DO ONNX RUNTIME:\n" << exception.what() << "\n";
        return 1;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "\nERRO:\n" << exception.what() << "\n";
        return 1;
    }
}
