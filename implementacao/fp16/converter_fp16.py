"""
Converte os modelos ONNX do SepConv para meia precisao (FP16).

As entradas (quadros e, no modelo em blocos, media e desvio) continuam em
FP32, para que o motor nao precise mudar o pre-processamento. As saidas (os
quatro tensores de filtros) passam a FP16, o que reduz a metade a memoria dos
filtros, alem da das ativacoes. No modelo do quadro inteiro, a media e o
desvio calculados dentro do grafo (ReduceMean, Pow, Sqrt) ficam em FP32: em
FP16, a soma de milhoes de pixels perderia precisao.

Uso:
    python implementacao/fp16/converter_fp16.py
"""

import os

import onnx
from onnxconverter_common import float16

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELOS = os.path.join(RAIZ, "modelos")

ENTRADAS = ["input_frame1", "input_frame2", "mean", "std"]
OPERACOES_EM_FP32 = float16.DEFAULT_OP_BLOCK_LIST + ["ReduceMean", "Pow", "Sqrt"]


def saidas_em_fp16(modelo):
    """O conversor devolve as saidas em FP32 com um Cast no fim; os Casts
    sao removidos para que os filtros saiam da rede em FP16."""
    produtores = {saida: no for no in modelo.graph.node for saida in no.output}

    for saida in modelo.graph.output:
        cast = produtores[saida.name]

        if cast.op_type != "Cast":
            continue

        anterior = produtores[cast.input[0]]
        anterior.output[list(anterior.output).index(cast.input[0])] = saida.name
        modelo.graph.node.remove(cast)
        saida.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16


def converter(nome):
    origem = os.path.join(MODELOS, f"{nome}.onnx")
    destino = os.path.join(MODELOS, f"{nome}_fp16.onnx")

    modelo = onnx.load(origem)
    entradas = [e.name for e in modelo.graph.input if e.name in ENTRADAS]

    convertido = float16.convert_float_to_float16(
        modelo,
        keep_io_types=entradas,
        op_block_list=OPERACOES_EM_FP32,
    )
    saidas_em_fp16(convertido)
    onnx.checker.check_model(convertido)
    onnx.save(convertido, destino)

    saidas = [(s.name, onnx.TensorProto.DataType.Name(s.type.tensor_type.elem_type)) for s in convertido.graph.output]
    print(f"{os.path.basename(destino)}: {os.path.getsize(destino) / 2**20:.1f} MiB, saidas {saidas}")


def main():
    for nome in ("sepconv_paper", "sepconv_blocos"):
        converter(nome)


if __name__ == "__main__":
    main()
