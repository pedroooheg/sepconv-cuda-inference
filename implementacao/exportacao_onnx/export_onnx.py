import os
import sys

# Raiz do projeto (dois niveis acima de implementacao/exportacao_onnx/)
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --blocos exporta a variante para processamento em blocos (ver SepConvTileExtractor)
EXPORTAR_BLOCOS = "--blocos" in sys.argv

sys.argv = sys.argv[:1]  # run.py faz getopt no import
sys.path.insert(0, os.path.join(RAIZ, "original"))

import torch
import torch.nn as nn
from run import Network, Decode


# O Decode original recorta a saida do upsampling com um "if" em Python que
# compara shapes. Durante o export esse "if" e avaliado uma unica vez e o
# recorte fica fixo no grafo, fazendo o ONNX funcionar apenas na resolucao do
# export. Aqui o recorte vira um slice para o shape do nivel atual, que e
# equivalente ao original e continua valido para qualquer resolucao par.
def decode_forward_exportavel(self, tenIns):
    intRow = self.intRows - 1
    for netHor in self.netHor:
        if self.intOuts[intRow] != 0 and self.intIns[intRow] != 0:
            tenIns[intRow] = netHor(tenIns[intRow])
        intRow -= 1

    intRow = self.intRows - 1
    for netVer in self.netVer:
        if self.intOuts[intRow] != 0 and intRow != self.intRows - 1:
            tenVer = netVer(tenIns[intRow + 1])
            tenVer = tenVer[:, :, :tenIns[intRow].shape[2], :tenIns[intRow].shape[3]]
            tenIns[intRow] = tenIns[intRow] + tenVer
        intRow -= 1

    return tenIns


Decode.forward = decode_forward_exportavel

# Classe wrapper para isolar apenas a inferência da CNN (sem a camada customizada sepconv_func)
class SepConvFeatureExtractor(nn.Module):
    def __init__(self, original_net):
        super().__init__()
        self.netInput = original_net.netInput
        self.netEncode = original_net.netEncode
        self.netDecode = original_net.netDecode
        self.netVerone = original_net.netVerone
        self.netVertwo = original_net.netVertwo
        self.netHorone = original_net.netHorone
        self.netHortwo = original_net.netHortwo
        self.intChannels = original_net.intChannels

    def forward(self, tenOne, tenTwo):
        # 1. Normalização estática dos quadros
        tenStats = [tenOne, tenTwo]
        tenMean = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
        tenStd = (sum([tenIn.std([1, 2, 3], False, True).square() + (tenMean - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()
        
        tenOneNorm = (tenOne - tenMean) / (tenStd + 0.0000001)
        tenTwoNorm = (tenTwo - tenMean) / (tenStd + 0.0000001)

        # 2. Extração de características (Encoder-Decoder)
        tenOut = self.netDecode(self.netEncode([torch.cat([self.netInput(tenOneNorm), self.netInput(tenTwoNorm)], 1)] + ([0.0] * (len(self.intChannels) - 1))))[1]

        # 3. Predição dos 4 vetores 1D (Vertical/Horizontal para Quadro 1 e Quadro 2)
        tenVerone = self.netVerone(tenOut)
        tenVertwo = self.netVertwo(tenOut)
        tenHorone = self.netHorone(tenOut)
        tenHortwo = self.netHortwo(tenOut)

        return tenVerone, tenVertwo, tenHorone, tenHortwo


# Variante para processamento em blocos: a media e o desvio padrao sao
# calculados pelo motor sobre o quadro inteiro e recebidos como entrada, para
# que todos os blocos sejam normalizados da mesma forma que o quadro completo.
class SepConvTileExtractor(SepConvFeatureExtractor):
    def forward(self, tenOne, tenTwo, tenMean, tenStd):
        tenOneNorm = (tenOne - tenMean) / (tenStd + 0.0000001)
        tenTwoNorm = (tenTwo - tenMean) / (tenStd + 0.0000001)

        tenOut = self.netDecode(self.netEncode([torch.cat([self.netInput(tenOneNorm), self.netInput(tenTwoNorm)], 1)] + ([0.0] * (len(self.intChannels) - 1))))[1]

        return self.netVerone(tenOut), self.netVertwo(tenOut), self.netHorone(tenOut), self.netHortwo(tenOut)


def export_sepconv_tiles_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SepConvTileExtractor(Network().to(device)).to(device)
    model.eval()

    # Bloco de 512 com margem de 96 px de cada lado
    height, width = 704, 704
    dummy_frame1 = torch.rand(1, 3, height, width, device=device)
    dummy_frame2 = torch.rand(1, 3, height, width, device=device)
    dummy_mean = torch.full((1, 1, 1, 1), 0.5, device=device)
    dummy_std = torch.full((1, 1, 1, 1), 0.25, device=device)

    output_onnx_path = os.path.join(RAIZ, "modelos", "sepconv_blocos.onnx")

    print("Exportando a variante para blocos...")
    torch.onnx.export(
        model,
        (dummy_frame1, dummy_frame2, dummy_mean, dummy_std),
        output_onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input_frame1', 'input_frame2', 'mean', 'std'],
        output_names=['kernel_v1', 'kernel_v2', 'kernel_h1', 'kernel_h2'],
        dynamic_axes={
            'input_frame1': {2: 'height', 3: 'width'},
            'input_frame2': {2: 'height', 3: 'width'},
            'kernel_v1': {2: 'height', 3: 'width'},
            'kernel_v2': {2: 'height', 3: 'width'},
            'kernel_h1': {2: 'height', 3: 'width'},
            'kernel_h2': {2: 'height', 3: 'width'}
        }
    )

    print(f"\nSucesso! Modelo ONNX salvo em: {os.path.abspath(output_onnx_path)}")


def export_sepconv_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizando dispositivo: {device}")

    # Instanciar e carregar a rede original
    full_model = Network().to(device)
    weights_path = os.path.join(RAIZ, 'modelos', 'network-paper.pytorch')
    
    if os.path.exists(weights_path):
        print(f"Carregando pesos de {weights_path}...")
        full_model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Criar o extrator isolado sem a camada C++ do sepconv_func
    model = SepConvFeatureExtractor(full_model).to(device)
    model.eval()

    # Criar dummy inputs (Exemplo: 1080p -> 1080x1920)
    height, width = 1080, 1920
    dummy_frame1 = torch.randn(1, 3, height, width, device=device)
    dummy_frame2 = torch.randn(1, 3, height, width, device=device)

    output_onnx_path = os.path.join(RAIZ, "modelos", "sepconv_paper.onnx")

    print("Exportando a sub-rede de filtros para ONNX...")
    torch.onnx.export(
        model,
        (dummy_frame1, dummy_frame2),
        output_onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input_frame1', 'input_frame2'],
        output_names=['kernel_v1', 'kernel_v2', 'kernel_h1', 'kernel_h2'],
        dynamic_axes={
            'input_frame1': {2: 'height', 3: 'width'},
            'input_frame2': {2: 'height', 3: 'width'},
            'kernel_v1': {2: 'height', 3: 'width'},
            'kernel_v2': {2: 'height', 3: 'width'},
            'kernel_h1': {2: 'height', 3: 'width'},
            'kernel_h2': {2: 'height', 3: 'width'}
        }
    )

    print(f"\nSucesso! Modelo ONNX salvo em: {os.path.abspath(output_onnx_path)}")

if __name__ == '__main__':
    if EXPORTAR_BLOCOS:
        export_sepconv_tiles_to_onnx()
    else:
        export_sepconv_to_onnx()