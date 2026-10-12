# Bateria de testes com os videos do GTA VI: C++ x Python.
#
# Para cada resolucao, cada implementacao interpola 2x o video de 15 FPS
# (mede desempenho e VRAM) e o resultado e comparado com o video real de
# 30 FPS (mede PSNR/SSIM/VMAF). Tudo fica em resultados\testes_gta\<data_hora>\.
#
# Uso (na raiz do projeto):
#   .\implementacao\avaliacao\executar_testes_gta.ps1
#   .\implementacao\avaliacao\executar_testes_gta.ps1 -Resolucoes 720p,1080p
#   .\implementacao\avaliacao\executar_testes_gta.ps1 -Incluir4K -QuadrosMax4K 5

param(
    [string[]] $Resolucoes = @("720p", "1080p", "1440p"),
    [switch] $Incluir4K,
    [int] $QuadrosMax4K = 5
)

$ErrorActionPreference = "Continue"

$Raiz = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Raiz

$Motor = "implementacao\motor_cpp\build\Release\sepconv_engine.exe"
$Python = "venv\Scripts\python.exe"
$Modelo = "modelos\sepconv_paper.onnx"
$Pasta = "resultados\testes_gta\" + (Get-Date -Format "yyyy-MM-dd_HH-mm")

foreach ($Arquivo in @($Motor, $Python, $Modelo)) {
    if (-not (Test-Path $Arquivo)) {
        Write-Host "Nao encontrado: $Arquivo" -ForegroundColor Red
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

    foreach ($Impl in @("cpp", "python")) {
        $Video = "$Pasta\saidas\${Impl}_${Res}.mkv"

        Write-Host "`n========== $Impl | $Res ==========" -ForegroundColor Cyan

        if ($Impl -eq "cpp") {
            & $Motor --model $Modelo --video $Entrada --out $Video --factor 2 --csv "$Pasta\desempenho.csv" @Extra
        }
        else {
            & $Python implementacao\avaliacao\benchmark_python.py --video $Entrada --out $Video --factor 2 --csv "$Pasta\desempenho.csv" @Extra
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "FALHOU: $Impl $Res" -ForegroundColor Red
            Add-Content "$Pasta\falhas.csv" "$Impl,$Res"
            continue
        }

        & $Python implementacao\avaliacao\avaliar_qualidade.py --referencia $Referencia --interpolado $Video --rotulo $Impl --csv "$Pasta\qualidade.csv" --csv-quadros "$Pasta\quadros\${Impl}_${Res}.csv"
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
