# Análise Experimental da Inferência SepConv em C++ com Ênfase em Eficiência Computacional e Uso de VRAM

Este repositório contém a implementação e os materiais computacionais desenvolvidos para o **Trabalho de Conclusão de Curso (TCC)** do curso de Bacharelado em Ciência da Computação do **Centro Universitário Senac — Campus Santo Amaro**.

O trabalho investiga experimentalmente a execução do modelo **SepConv (Separable Convolution)** para **Interpolação de Quadros de Vídeo (VFI — Video Frame Interpolation)**, com foco na análise de **latência de execução, desempenho computacional e utilização de memória de vídeo (VRAM)**.

A implementação proposta utiliza **C++17, CUDA e ONNX Runtime**, buscando avaliar o comportamento da inferência em uma GPU de consumo com capacidade limitada de memória.

---

## 📚 Sobre o Trabalho

A interpolação de quadros de vídeo consiste na síntese de quadros intermediários a partir de quadros consecutivos de um vídeo. Essa técnica pode ser utilizada, por exemplo, para aumentar a taxa de quadros e melhorar a fluidez temporal de conteúdos audiovisuais.

Neste trabalho, é investigada a abordagem de **Convoluções Adaptativas Separáveis (SepConv)** proposta por Niklaus, Mai e Wang.

Diferentemente de abordagens baseadas diretamente em fluxo óptico denso, a SepConv trata a síntese do quadro intermediário como um processo de filtragem local adaptativa. Os kernels bidimensionais utilizados na filtragem podem ser decompostos em filtros unidimensionais, reduzindo o número de operações necessárias para a aplicação da convolução.

O objetivo desta pesquisa não é realizar treinamento ou desenvolver um novo modelo de aprendizado de máquina. O foco está na **implementação e análise experimental da etapa de inferência**.

---

## 🎯 Objetivo

O objetivo principal é analisar experimentalmente a inferência do modelo SepConv em uma implementação baseada em **C++ e CUDA**, avaliando seu comportamento em relação à eficiência computacional e ao uso de memória de vídeo.

A análise considera principalmente:

* **Latência de execução**;
* **Desempenho em FPS (Frames Per Second)**;
* **Consumo e pico de utilização de VRAM**;
* Comportamento da execução das convoluções na GPU;
* Qualidade dos quadros interpolados.

A implementação também permite comparar o comportamento da abordagem desenvolvida com a implementação original do modelo.

---

## 🧩 Abordagem da Implementação

A implementação separa o processo de inferência em componentes responsáveis por diferentes etapas do processamento.

```text
                 Quadros de entrada
                         │
                         ▼
                ┌─────────────────┐
                │     OpenCV      │
                │ Leitura do vídeo│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Preparação dos  │
                │      dados      │
                └────────┬────────┘
                         │
                  Host → Device
                         │
                         ▼
                ┌─────────────────┐
                │      GPU        │
                │      VRAM       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  ONNX Runtime   │
                │ CUDA Provider   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  CUDA Kernels   │
                │  SepConv        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Quadro gerado   │
                │     t + 0.5     │
                └────────┬────────┘
                         │
                  Device → Host
                         │
                         ▼
                ┌─────────────────┐
                │     OpenCV      │
                │ Gravação vídeo  │
                └─────────────────┘
```

### Etapas principais

**1. Entrada dos dados**

Os quadros consecutivos do vídeo são lidos utilizando a API C++ do OpenCV.

**2. Preparação dos quadros**

Os dados dos quadros são organizados em memória para posterior transferência e processamento na GPU.

**3. Transferência para a GPU**

Os dados são transferidos da memória principal (RAM) para a memória da GPU (VRAM).

**4. Inferência do modelo**

O modelo convertido para o formato **ONNX** é executado utilizando o **ONNX Runtime com CUDA Execution Provider**.

**5. Aplicação da convolução separável**

Os resultados da inferência são utilizados pelos kernels CUDA responsáveis pela aplicação das operações de convolução separável na GPU.

**6. Geração do quadro intermediário**

A aplicação das filtragens produz o quadro intermediário correspondente ao instante temporal entre os dois quadros de entrada.

**7. Saída**

O quadro gerado é transferido para a memória principal e posteriormente utilizado para a composição do vídeo de saída.

---

## 🛠️ Tecnologias

| Tecnologia                  | Função no projeto                          |
| --------------------------- | ------------------------------------------ |
| **C++17**                   | Implementação principal                    |
| **CUDA C++**                | Processamento paralelo na GPU              |
| **ONNX Runtime**            | Execução do modelo neural                  |
| **CUDA Execution Provider** | Execução do modelo utilizando a GPU        |
| **OpenCV**                  | Leitura, processamento e gravação de vídeo |
| **NVML**                    | Monitoramento da utilização de VRAM        |
| **CUDA Events**             | Medição de tempos de execução na GPU       |
| **CMake**                   | Sistema de compilação                      |

---

## 💻 Ambiente Experimental

Os experimentos são realizados em uma GPU NVIDIA de consumo com **8 GB de VRAM**.

### Hardware utilizado

* **GPU:** NVIDIA GeForce RTX 4060 Ti
* **VRAM:** 8 GB

O foco em uma GPU com 8 GB está relacionado ao escopo definido no TCC, que investiga o comportamento da inferência em um cenário de hardware comercial com capacidade de memória limitada.

> As especificações completas do ambiente experimental, incluindo versões de software e configurações utilizadas em cada experimento, serão registradas junto aos resultados obtidos.

---

## 📦 Modelo Utilizado

O projeto utiliza como base o modelo pré-treinado desenvolvido por:

**Simon Niklaus, Long Mai e Oliver Wang**

O trabalho original é:

> Niklaus, S.; Mai, L.; Wang, O. *Revisiting Adaptive Convolutions for Video Frame Interpolation*. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, p. 1099–1109.

### Repositório original

https://github.com/sniklaus/revisiting-sepconv

O modelo e os trabalhos dos autores originais são utilizados como referência para a implementação experimental apresentada neste repositório.

---

## 🔄 Modelo ONNX

Para permitir a execução através da API C++ do ONNX Runtime, o modelo originalmente disponibilizado em formato compatível com PyTorch é utilizado como base para a conversão para **ONNX**.

O modelo original disponibiliza, entre outros arquivos, os pesos utilizados na rede:

```text
network-paper.pytorch
```

A exportação é feita por `implementacao/exportacao_onnx/export_onnx.py`, que gera `modelos/sepconv_paper.onnx` (+ `.onnx.data`). O script `validar_onnx.py`, na mesma pasta, compara as saídas do ONNX com as do PyTorch em várias resoluções.

> O `Decode` original recorta tensores com um `if` em Python dependente do tamanho da imagem. No export esse recorte ficava fixo, e o ONNX só funcionava em 1080×1920. O `export_onnx.py` substitui o recorte por um *slice* equivalente, válido para qualquer resolução. A versão antiga está em `modelos/onnx_antigo_so_1080p/`.

> Os arquivos de pesos e modelos originais devem ser utilizados de acordo com as condições de licença e distribuição estabelecidas pelos respectivos autores.

---

## ⚙️ Pré-requisitos

Para compilar e executar o projeto, são necessários:

* GPU NVIDIA compatível com CUDA;
* NVIDIA CUDA Toolkit;
* compilador compatível com C++17;
* CMake;
* OpenCV para C++;
* ONNX Runtime para C++;
* ONNX Runtime com suporte ao CUDA Execution Provider.

### Versões

As versões específicas utilizadas nos experimentos devem ser consultadas na configuração do ambiente e registradas para garantir a reprodutibilidade dos experimentos.

---

## 🚀 Compilação

O motor fica em `implementacao/motor_cpp/`. O `CMakePresets.json` já define o gerador (Visual Studio 2022) e o toolset CUDA 13.3:

```bash
cd implementacao/motor_cpp
cmake --preset release
cmake --build --preset release
```

Os executáveis são gerados em `implementacao/motor_cpp/build/Release/`.

> O ONNX Runtime 1.23 (GPU) depende das DLLs do CUDA 12 e do cuDNN 9. Por padrão o motor usa as que acompanham o PyTorch em `venv/Lib/site-packages/torch/lib`; outro diretório pode ser indicado pela variável de ambiente `SEPCONV_CUDA_DLL_DIR`.

---

## ▶️ Execução

A partir da raiz do projeto:

```powershell
.\implementacao\motor_cpp\build\Release\sepconv_engine.exe `
    --model modelos\sepconv_paper.onnx `
    --one original\images\one.png `
    --two original\images\two.png `
    --out resultados\validacao_cpp\cpp_one_two.png `
    --runs 20 --warmup 5
```

Opções do CUDA Execution Provider podem ser repassadas com `--ort-option chave=valor`.

Processamento em blocos (v2): `--bloco 512 --margem 96`, com o modelo `modelos\sepconv_blocos.onnx` (gerado por `export_onnx.py --blocos`). A margem de 96 px foi definida pelo estudo em `implementacao\tiling\estudo_margem.py`. Com `--bloco 0` (padrão), o motor processa o quadro inteiro, como na v1.

Baseline em Python e comparação de qualidade:

```powershell
venv\Scripts\python.exe implementacao\avaliacao\benchmark_python.py --one original\images\one.png --two original\images\two.png
venv\Scripts\python.exe implementacao\avaliacao\comparar_imagens.py resultados\validacao_cpp\python_one_two.png resultados\validacao_cpp\cpp_one_two.png
```

---

## 🎮 Testes com os vídeos do GTA VI

1. Gerar os vídeos de teste (trecho de 8 s a 10 s, sem cortes de cena, em 720p/1080p/1440p/2160p, sem perdas):
   `venv/Scripts/python.exe implementacao/utilitarios_video/preparar_testes.py`
2. Executar a bateria C++ × Python (desempenho, VRAM, PSNR/SSIM):
   `./implementacao/avaliacao/executar_testes_gta.ps1` (adicione `-Incluir4K` para o 2160p)
3. Os resultados ficam em `resultados/testes_gta/<data_hora>/`: `desempenho.csv`, `qualidade.csv` (PSNR, SSIM e VMAF apenas dos quadros sintetizados), `quadros/` (métricas por quadro), `resumo.md`, `log.txt` e os vídeos em `saidas/`.
4. Os gráficos (PDF + PNG) e a tabela LaTeX são gerados automaticamente em `graficos/`. Para refazê-los:
   `venv/Scripts/python.exe implementacao/avaliacao/gerar_graficos.py resultados/testes_gta/<data_hora>`

> O VMAF exige o FFmpeg com `libvmaf` no PATH.

---

## 📊 Avaliação Experimental

A avaliação do projeto está estruturada em três aspectos principais.

### Desempenho computacional

São analisados:

* tempo de execução;
* latência por quadro;
* FPS obtidos;
* comportamento da execução na GPU.

### Utilização de memória

A utilização da memória de vídeo é monitorada para avaliar:

* consumo de VRAM;
* pico de utilização;
* comportamento da memória durante a inferência.

O monitoramento é realizado utilizando a **NVIDIA Management Library (NVML)**.

### Qualidade da interpolação

A qualidade dos quadros produzidos pode ser avaliada utilizando métricas objetivas como:

* **PSNR (Peak Signal-to-Noise Ratio)**;
* **SSIM (Structural Similarity Index)**.

Também pode ser realizada análise visual dos resultados para identificação de artefatos, como:

* *ghosting*;
* desfoque;
* distorções;
* inconsistências em regiões com movimento.

---

## 🧪 Comparação Experimental

A implementação desenvolvida neste repositório será utilizada para comparação com a implementação original do modelo.

A comparação considera principalmente:

```text
                 Implementações
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    Implementação              Implementação
       original                   em C++
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
              Avaliação experimental
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Latência       FPS         VRAM
                       │
                       ▼
                 Qualidade
              PSNR / SSIM
```

Os resultados quantitativos serão apresentados após a realização dos experimentos definidos na metodologia do TCC.

---

## 📁 Estrutura do Repositório

```text
.
├── README.md
├── original/                  Trabalho original (Niklaus) — código de referência
│   ├── run.py, sepconv.py     Rede e camada sepconv em PyTorch (baseline)
│   ├── benchmark.py, LICENSE, requirements.txt
│   ├── images/                Par de quadros de exemplo (one.png / two.png)
│   └── middlebury/            Benchmark Middlebury (com ground truth)
├── implementacao/             Implementação deste TCC
│   ├── motor_cpp/             Motor C++/CUDA (CMake, src/, include/)
│   ├── exportacao_onnx/       Exportação PyTorch → ONNX e validação
│   ├── avaliacao/             Baseline Python e métricas (PSNR/SSIM)
│   └── utilitarios_video/     Scripts de preparo e montagem de vídeos
├── modelos/                   Pesos .pytorch e modelos .onnx
├── dados/
│   └── videos/                Vídeos de entrada (GTA VI, Lucia, car-turn)
├── resultados/                Saídas geradas
│   ├── videos/                Vídeos interpolados e comparações
│   ├── quadros_gta_838_957/   Quadros extraídos para análise
│   ├── figuras_tcc/           Imagens usadas no texto do TCC
│   └── validacao_cpp/         Saídas C++ × Python usadas na validação
├── tcc/                       Aviso: o texto fica em github.com/pedroooheg/video-frame-interpolation
└── venv/                      Ambiente Python (não versionado)
```

> Em `original/run.py`, a única alteração em relação ao repositório de Niklaus é a desativação das asserções que limitavam a resolução de entrada a 1280×720.

---

## 📈 Resultados

Os resultados experimentais serão adicionados ao repositório após a conclusão da etapa de implementação e execução dos testes.

Entre os resultados previstos estão:

* comparação de latência;
* comparação de FPS;
* utilização de VRAM;
* análise de qualidade dos quadros;
* comparação visual entre os resultados.

Esta seção será atualizada conforme os experimentos forem concluídos.

---

## 📖 Referências

### Trabalho principal

NIKLAUS, Simon; MAI, Long; WANG, Oliver. **Revisiting Adaptive Convolutions for Video Frame Interpolation**. In: *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*. 2021. p. 1099–1109.

### Repositório original

Niklaus, Simon. **Revisiting Adaptive Convolutions for Video Frame Interpolation**.

https://github.com/sniklaus/revisiting-sepconv

---

## 🎓 Informações Acadêmicas

**Instituição:** Centro Universitário Senac — Campus Santo Amaro
**Curso:** Bacharelado em Ciência da Computação
**Autor:** Pedro Henrique Gomes
**Orientador:** Prof. Afonso Brandão
**Ano:** 2026

### Título do TCC

**Análise Experimental da Inferência SepConv em C++ com Ênfase em Eficiência Computacional e Uso de VRAM**

---

## ⚠️ Status

Este repositório está em desenvolvimento e integra o Trabalho de Conclusão de Curso.

A implementação e os experimentos podem sofrer alterações durante o desenvolvimento do projeto. Os resultados apresentados devem ser interpretados considerando as condições experimentais e configurações de hardware e software utilizadas.

---
