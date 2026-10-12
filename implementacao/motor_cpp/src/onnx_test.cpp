#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << "Teste ONNX Runtime - C++\n";
    std::cout << "========================================\n\n";

    // Caminho do modelo.
    // Pode ser passado como primeiro argumento.
       // Caminho do modelo.
    // No Windows, usamos std::wstring para a API do ONNX Runtime.
    std::wstring model_path;

    if (argc >= 2)
    {
        // Converte o argumento recebido para uma string wide.
        std::string input_path = argv[1];

        model_path.assign(
            input_path.begin(),
            input_path.end()
        );
    }
    else
    {
        std::cerr << "Erro: caminho do modelo nao informado.\n";
        std::cerr << "Uso:\n";
        std::cerr << "onnx_test.exe \"caminho\\para\\sepconv_paper.onnx\"\n";
        return 1;
    }

    std::cout << "Modelo: " << argv[1] << "\n\n";

    try
    {
        // Inicializa o ambiente do ONNX Runtime.
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "SepConvTest"
        );

        std::cout << "ONNX Runtime inicializado.\n";

        // Configuracoes da sessao.
        Ort::SessionOptions session_options;

        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        // Cria a sessao e carrega o modelo.
        Ort::Session session(
            env,
            model_path.c_str(),
            session_options
        );

        std::cout << "Modelo carregado com sucesso!\n\n";

        Ort::AllocatorWithDefaultOptions allocator;

        // =========================================================
        // ENTRADAS
        // =========================================================

        std::cout << "ENTRADAS:\n";

        const size_t input_count =
            session.GetInputCount();

        for (size_t i = 0; i < input_count; ++i)
        {
            auto input_name =
                session.GetInputNameAllocated(i, allocator);

            auto input_type_info =
                session.GetInputTypeInfo(i);

            auto tensor_info =
                input_type_info.GetTensorTypeAndShapeInfo();

            auto shape =
                tensor_info.GetShape();

            std::cout << "  [" << i << "] "
                      << input_name.get()
                      << " | shape: ";

            for (auto dimension : shape)
            {
                std::cout << dimension << " ";
            }

            std::cout << "\n";
        }

        // =========================================================
        // SAIDAS
        // =========================================================

        std::cout << "\nSAIDAS:\n";

        const size_t output_count =
            session.GetOutputCount();

        for (size_t i = 0; i < output_count; ++i)
        {
            auto output_name =
                session.GetOutputNameAllocated(i, allocator);

            auto output_type_info =
                session.GetOutputTypeInfo(i);

            auto tensor_info =
                output_type_info.GetTensorTypeAndShapeInfo();

            auto shape =
                tensor_info.GetShape();

            std::cout << "  [" << i << "] "
                      << output_name.get()
                      << " | shape: ";

            for (auto dimension : shape)
            {
                std::cout << dimension << " ";
            }

            std::cout << "\n";
        }

        std::cout << "\n========================================\n";
        std::cout << "TESTE CONCLUIDO COM SUCESSO!\n";
        std::cout << "========================================\n";

        return 0;
    }
    catch (const Ort::Exception& exception)
    {
        std::cerr << "\nERRO DO ONNX RUNTIME:\n";
        std::cerr << exception.what() << "\n";
        return 1;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "\nERRO:\n";
        std::cerr << exception.what() << "\n";
        return 1;
    }
}