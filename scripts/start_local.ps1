param(
    [ValidateRange(1, 120)]
    [int]$HealthTimeoutSeconds = 30,
    [switch]$Headless
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$backendRoot = Join-Path $repositoryRoot 'backend'
$frontendRoot = Join-Path $repositoryRoot 'frontend'
$resolver = Join-Path $PSScriptRoot 'resolve_runtime.ps1'
$backendEntry = Join-Path $backendRoot 'app\run_server.py'
$frontendEntry = Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'
$started = New-Object System.Collections.Generic.List[System.Diagnostics.Process]

function Stop-StartedProcesses {
    foreach ($process in $started) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-HttpEndpoint([string]$url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Wait-ForEndpoint(
    [string]$component,
    [string]$url,
    [System.Diagnostics.Process]$process
) {
    $deadline = [DateTime]::UtcNow.AddSeconds($HealthTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($process.HasExited) {
            throw "$component process exited before becoming healthy."
        }
        if (Test-HttpEndpoint $url) {
            Write-Output "$component ready."
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "$component did not become healthy within the startup timeout."
}

try {
    if (-not (Test-Path -LiteralPath $backendEntry -PathType Leaf)) {
        throw 'Backend entry point is missing.'
    }
    if (-not (Test-Path -LiteralPath $frontendEntry -PathType Leaf)) {
        throw 'Frontend dependencies are missing. Install the declared frontend dependencies first.'
    }

    $python = & $resolver -Kind Python
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($python)) {
        throw 'Python runtime discovery failed.'
    }
    $node = & $resolver -Kind Node
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($node)) {
        throw 'Node.js runtime discovery failed.'
    }

    $startOptions = @{}
    if ($Headless) { $startOptions['NoNewWindow'] = $true }

    $env:BACKEND_RELOAD = 'false'
    $backend = Start-Process -FilePath $python -ArgumentList @('-m', 'app.run_server') `
        -WorkingDirectory $backendRoot -PassThru @startOptions
    $started.Add($backend)
    Write-Output 'Backend process started; waiting for health check.'
    Wait-ForEndpoint 'Backend' 'http://127.0.0.1:8000/api/health' $backend

    $frontend = Start-Process -FilePath $node `
        -ArgumentList @('node_modules/vite/bin/vite.js', '--configLoader', 'native', '--host', '127.0.0.1', '--port', '5173') `
        -WorkingDirectory $frontendRoot -PassThru @startOptions
    $started.Add($frontend)
    Write-Output 'Frontend process started; waiting for HTTP response.'
    Wait-ForEndpoint 'Frontend' 'http://127.0.0.1:5173/' $frontend

    Write-Output 'RepoNoesis backend and frontend are ready.'
    Write-Output 'Backend:  http://127.0.0.1:8000'
    Write-Output 'Frontend: http://127.0.0.1:5173'
    exit 0
} catch {
    Stop-StartedProcesses
    [Console]::Error.WriteLine("RepoNoesis startup failed: $($_.Exception.Message)")
    exit 1
}
