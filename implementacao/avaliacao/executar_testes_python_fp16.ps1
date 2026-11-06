# Bateria 2 x 2: implementacao (Python, C++) x precisao (FP32, FP16).
#
# Mede as quatro combinacoes nas mesmas condicoes, com o quadro inteiro, para
# separar o efeito da forma de execucao do efeito da precisao. Segue o mesmo
# protocolo de executar_testes_gta.ps1. Tudo fica em
# resultados\testes_python_fp16\<data_hora>\.
#
# Uso (na raiz do projeto):
#   .\implementacao\avaliacao\executar_testes_python_fp16.ps1
#   .\implementacao\avaliacao\executar_testes_python_fp16.ps1 -Resolucoes 720p,1080p -Sem4K

param(
    [string[]] $Resolucoes = @("720p", "1080p", "1440p"),
    [switch] $Sem4K,
    [int] $QuadrosMax4K = 5
)

$ErrorActionPreference = "Continue"

$Raiz = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Raiz

$Motor = "implementacao\motor_cpp\build\Release\sepconv_engine.exe"
$Python = "venv\Scripts\python.exe"
$Pasta = "resultados\testes_python_fp16\" + (Get-Date -Format "yyyy-MM-dd_HH-mm")
$Series = @("python", "python_fp16", "cpp", "cpp_fp16")

foreach ($Arquivo in @($Motor, $Python, "modelos\sepconv_paper.onnx", "modelos\sepconv_paper_fp16.onnx", "dados\testes\gta_720p_15fps.mkv")) {
    if (-not (Test-Path $Arquivo)) {
        Write-Host "Nao encontrado: $Arquivo" -ForegroundColor Red
        exit 1
    }
}

New-Item -ItemType Directory -Force "$Pasta\saidas" | Out-Null
Start-Transcript -Path "$Pasta\log.txt" | Out-Null

# Estado da GPU no inicio da bateria.
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu,clocks.sm --format=csv |
    Out-File "$Pasta\ambiente.txt"

function Testar([string] $Res, [int] $MaxQuadros) {
    $Entrada = "dados\testes\gta_${Res}_15fps.mkv"
    $Referencia = "dados\testes\gta_${Res}_30fps.mkv"
    $Csv = "$Pasta\desempenho.csv"

    $Extra = @()
    if ($MaxQuadros -gt 0) { $Extra = @("--max-frames", "$MaxQuadros", "--warmup", "1") }

    foreach ($Serie in $Series) {
        $Video = "$Pasta\saidas\${Serie}_${Res}.mkv"

        Write-Host "`n========== $Serie | $Res ==========" -ForegroundColor Cyan

        switch ($Serie) {
            "python" { & $Python implementacao\avaliacao\benchmark_python.py --video $Entrada --out $Video --factor 2 --csv $Csv @Extra }
            "python_fp16" { & $Python implementacao\avaliacao\benchmark_python_fp16.py --video $Entrada --out $Video --factor 2 --csv $Csv @Extra }
            "cpp" { & $Motor --model modelos\sepconv_paper.onnx --video $Entrada --out $Video --factor 2 --csv $Csv @Extra }
            "cpp_fp16" { & $Motor --model modelos\sepconv_paper_fp16.onnx --video $Entrada --out $Video --factor 2 --csv $Csv @Extra }
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

if (-not $Sem4K) {
    Testar "2160p" $QuadrosMax4K
}

& $Python implementacao\avaliacao\resumir_testes.py $Pasta
& $Python implementacao\avaliacao\analisar_python_fp16.py $Pasta

Stop-Transcript | Out-Null
Write-Host "`nResultados em: $Pasta" -ForegroundColor Green
