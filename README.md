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

O arquivo ONNX utilizado pela implementação deve estar disponível no diretório definido pela configuração do projeto.

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

Clone o repositório e entre no diretório do projeto:

```bash
git clone <URL_DO_REPOSITORIO>
cd <DIRETORIO_DO_REPOSITORIO>
```

Crie o diretório de compilação:

```bash
mkdir build
cd build
```

Configure o projeto com o CMake:

```bash
cmake ..
```

Compile em modo Release:

```bash
cmake --build . --config Release
```

> Os comandos podem variar de acordo com o sistema operacional, compilador e gerador utilizado pelo CMake.

---

## ▶️ Execução

Após a compilação, a aplicação pode ser executada informando o vídeo de entrada, o modelo ONNX e o caminho do vídeo de saída.

Exemplo:

```bash
./sepconv_cuda \
    --video input.mp4 \
    --model sepconv_paper.onnx \
    --output output.mp4
```

No Windows, utilizando um executável gerado pelo Visual Studio:

```powershell
.\Release\sepconv_cuda.exe `
    --video input.mp4 `
    --model sepconv_paper.onnx `
    --output output.mp4
```

Os parâmetros apresentados dependem da interface de linha de comando implementada no projeto.

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

A organização final do repositório pode seguir a estrutura:

```text
.
├── CMakeLists.txt
├── README.md
├── include/
├── src/
├── kernels/
├── models/
├── videos/
└── build/
```

A estrutura pode ser modificada durante o desenvolvimento conforme a organização definitiva dos componentes da implementação.

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
