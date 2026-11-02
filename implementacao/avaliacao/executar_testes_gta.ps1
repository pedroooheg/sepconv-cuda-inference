# Bateria de testes com os videos do GTA VI: C++ x Python.
#
# Para cada resolucao, cada serie de -Series interpola 2x o video de 15 FPS
# (mede desempenho e VRAM) e o resultado e comparado com o video real de
# 30 FPS (mede PSNR/SSIM/VMAF). Tudo fica em resultados\testes_gta\<data_hora>\.
#
# Series (os mesmos rotulos que o motor grava no CSV):
#   python             implementacao original em PyTorch
#   cpp                motor C++ com o quadro inteiro
#   cpp_b<T>           motor C++ em blocos T x T (margem -Margem)
#   sufixo _fp16       modelo em meia precisao (cpp_fp16, cpp_b1024_fp16, ...)
#
# Uso (na raiz do projeto):
#   .\implementacao\avaliacao\executar_testes_gta.ps1
#   .\implementacao\avaliacao\executar_testes_gta.ps1 -Resolucoes 720p,1080p
#   .\implementacao\avaliacao\executar_testes_gta.ps1 -Incluir4K -QuadrosMax4K 5
#   .\implementacao\avaliacao\executar_testes_gta.ps1 -Series cpp,cpp_b1024,cpp_b512,cpp_b256,python   (bateria v2)

param(
    [string[]] $Resolucoes = @("720p", "1080p", "1440p"),
    [switch] $Incluir4K,
    [int] $QuadrosMax4K = 5,
    [string[]] $Series = @("cpp", "cpp_fp16", "cpp_b1024_fp16", "cpp_b512_fp16", "python"),
    [int] $Margem = 96
)

$ErrorActionPreference = "Continue"

$Raiz = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Raiz

$Motor = "implementacao\motor_cpp\build\Release\sepconv_engine.exe"
$Python = "venv\Scripts\python.exe"
$Pasta = "resultados\testes_gta\" + (Get-Date -Format "yyyy-MM-dd_HH-mm")

# Modelo e argumentos do motor para cada serie C++.
function Configuracao([string] $Serie) {
    if ($Serie -notmatch '^cpp(_b(\d+))?(_fp16)?$') {
        throw "Serie desconhecida: $Serie"
    }

    $Bloco = if ($Matches[2]) { [int] $Matches[2] } else { 0 }
    $Nome = if ($Bloco -gt 0) { "sepconv_blocos" } else { "sepconv_paper" }
    $Sufixo = if ($Matches[3]) { "_fp16" } else { "" }
    $Argumentos = if ($Bloco -gt 0) { @("--bloco", "$Bloco", "--margem", "$Margem") } else { @() }

    return @{ Modelo = "modelos\$Nome$Sufixo.onnx"; Argumentos = $Argumentos }
}

$Necessarios = @($Motor, $Python) + @($Series | Where-Object { $_ -ne "python" } | ForEach-Object { (Configuracao $_).Modelo })

foreach ($Arquivo in ($Necessarios | Select-Object -Unique)) {
    if (-not (Test-Path $Arquivo)) {
        Write-Host "Nao encontrado: $Arquivo" -ForegroundColor Red
        Write-Host "  (os modelos FP16 sao gerados por implementacao\fp16\converter_fp16.py)"
        exit 1
    }
}

if (-not (Test-Path "dados\testes\gta_720p_15fps.mkv")) {
    Write-Host "Videos de teste ausentes. Rode antes:" -ForegroundColor Red
    Write-Host "  venv\Scripts\python.exe implementacao\utilitarios_video\preparar_testes.py"
    exit 1
}

New-Item -ItemType Directory -Force "$Pasta\saidas" | Out-Null
Start-Transcript -Path "$Pasta\log.txt" | Out-Null

# Estado da GPU no inicio da bateria.
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu,clocks.sm --format=csv |
    Out-File "$Pasta\ambiente.txt"

function Testar([string] $Res, [int] $MaxQuadros) {
    $Entrada = "dados\testes\gta_${Res}_15fps.mkv"
    $Referencia = "dados\testes\gta_${Res}_30fps.mkv"

    $Extra = @()
    # Com poucos quadros (4K), so 1 interpolacao de aquecimento, para sobrarem
    # medicoes suficientes.
    if ($MaxQuadros -gt 0) { $Extra = @("--max-frames", "$MaxQuadros", "--warmup", "1") }

    foreach ($Serie in $Series) {
        $Video = "$Pasta\saidas\${Serie}_${Res}.mkv"

        Write-Host "`n========== $Serie | $Res ==========" -ForegroundColor Cyan

        if ($Serie -eq "python") {
            & $Python implementacao\avaliacao\benchmark_python.py --video $Entrada --out $Video --factor 2 --csv "$Pasta\desempenho.csv" @Extra
        }
        else {
            $Config = Configuracao $Serie
            $Argumentos = $Config.Argumentos
            & $Motor --model $Config.Modelo --video $Entrada --out $Video --factor 2 --csv "$Pasta\desempenho.csv" @Argumentos @Extra
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "FALHOU: $Serie $Res" -ForegroundColor Red
            Add-Content "$Pasta\falhas.csv" "$Serie,$Res"
            continue
        }

        & $Python implementacao\avaliacao\avaliar_qualidade.py --referencia $Referencia --interpolado $Video --rotulo $Serie --csv "$Pasta\qualidade.csv" --csv-quadros "$Pasta\quadros\${Serie}_${Res}.csv"
    }
}

foreach ($Res in $Resolucoes) {
    Testar $Res 0
}

if ($Incluir4K) {
    Testar "2160p" $QuadrosMax4K
}

& $Python implementacao\avaliacao\resumir_testes.py $Pasta
& $Python implementacao\avaliacao\gerar_graficos.py $Pasta

Stop-Transcript | Out-Null
Write-Host "`nResultados em: $Pasta" -ForegroundColor Green
