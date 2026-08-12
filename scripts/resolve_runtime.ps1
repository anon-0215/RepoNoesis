param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Python', 'Node')]
    [string]$Kind
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Resolve-ConfiguredExecutable([string]$environmentName) {
    $configured = [Environment]::GetEnvironmentVariable($environmentName, 'Process')
    if ([string]::IsNullOrWhiteSpace($configured)) { return $null }
    if (-not (Test-Path -LiteralPath $configured -PathType Leaf)) {
        [Console]::Error.WriteLine("Configured $environmentName executable is unavailable.")
        exit 2
    }
    return (Resolve-Path -LiteralPath $configured).Path
}

if ($Kind -eq 'Python') {
    $candidate = Resolve-ConfiguredExecutable 'REPONOESIS_PYTHON'
    if ($candidate) { Write-Output $candidate; exit 0 }

    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $condaPython = Join-Path $env:CONDA_PREFIX 'python.exe'
        if (Test-Path -LiteralPath $condaPython -PathType Leaf) {
            Write-Output (Resolve-Path -LiteralPath $condaPython).Path
            exit 0
        }
    }

    $command = Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) { Write-Output $command.Source; exit 0 }

    $repositoryPython = Join-Path $repositoryRoot 'backend\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $repositoryPython -PathType Leaf) {
        Write-Output (Resolve-Path -LiteralPath $repositoryPython).Path
        exit 0
    }

    [Console]::Error.WriteLine(
        'Cannot find Python. Activate the intended Conda environment or set REPONOESIS_PYTHON.'
    )
    exit 3
}

$candidate = Resolve-ConfiguredExecutable 'REPONOESIS_NODE'
if ($candidate) { Write-Output $candidate; exit 0 }

$command = Get-Command node -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($command) { Write-Output $command.Source; exit 0 }

$repositoryNode = Join-Path $repositoryRoot 'frontend\.node\node.exe'
if (Test-Path -LiteralPath $repositoryNode -PathType Leaf) {
    Write-Output (Resolve-Path -LiteralPath $repositoryNode).Path
    exit 0
}

[Console]::Error.WriteLine(
    'Cannot find Node.js. Install Node.js or set REPONOESIS_NODE.'
)
exit 3
