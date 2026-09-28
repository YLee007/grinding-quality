param(
    [ValidateSet('analyze','camera','annotate','train-rf','train-unet','export','self-test')]
    [string]$Task = 'analyze',
    [string]$Config = 'config/quality.yaml'
)
$ErrorActionPreference = 'Stop'
$qualityRoot = $PSScriptRoot
Push-Location -LiteralPath $qualityRoot
try {
    & "$qualityRoot\.venv\Scripts\python.exe" "$qualityRoot\quality.py" $Task --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Quality task failed, exit code $LASTEXITCODE" }
} finally { Pop-Location }
