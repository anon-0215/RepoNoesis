param(
    [ValidateRange(1, 120)][int]$HealthTimeoutSeconds = 30,
    [ValidateRange(1, 65535)][int]$BackendPort = 8000,
    [ValidateRange(1, 65535)][int]$FrontendPort = 5173,
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
$launchId = [Guid]::NewGuid().ToString('N')
$backendUrl = "http://127.0.0.1:$BackendPort"
$frontendUrl = "http://127.0.0.1:$FrontendPort"

function Stop-StartedProcesses {
    foreach ($process in $started) {
        if (-not $process.HasExited) {
            # Only roots created by this invocation, including their own children.
            $previousAction = $ErrorActionPreference
            try {
                $ErrorActionPreference = "Continue"
                & "$env:SystemRoot\System32\taskkill.exe" /PID $process.Id /T /F 2>&1 | Out-Null
            } finally { $ErrorActionPreference = $previousAction }
            $null = $process.WaitForExit(3000)
        }
    }
}
function Assert-FreePort([string]$component, [int]$port) {
    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $port)
    try {
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
    } catch {
        throw "$component port 127.0.0.1:$port is occupied or unavailable. Keep the existing service; close your own confirmed service or select free -BackendPort/-FrontendPort values."
    } finally { $listener.Stop() }
}
function Test-HttpEndpoint([string]$component,[string]$url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2
        if ($response.StatusCode -ne 200) { return $false }
        if ($component -eq 'Backend') {
            $body = $response.Content | ConvertFrom-Json
            return $body.startup_instance_id -ceq $launchId
        }
        return $response.Headers['X-RepoNoesis-Startup-Id'] -ceq $launchId
    } catch { return $false }
}
function Wait-ForEndpoint([string]$component,[string]$url,[System.Diagnostics.Process]$process) {
    $deadline = [DateTime]::UtcNow.AddSeconds($HealthTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($process.HasExited) { throw "$component process exited before becoming healthy." }
        $matches = Test-HttpEndpoint $component $url
        # A response can race process exit. Both the instance and child must match.
        if ($process.HasExited) { throw "$component process exited before becoming healthy." }
        if ($matches) { Write-Output "$component ready (this launch; PID $($process.Id))."; return }
        Start-Sleep -Milliseconds 250
    }
    throw "$component did not become healthy within the startup timeout; the endpoint must belong to this launch. Check the child log and port $url."
}
try {
    if (-not (Test-Path -LiteralPath $backendEntry -PathType Leaf)) { throw 'Backend entry point is missing.' }
    if (-not (Test-Path -LiteralPath $frontendEntry -PathType Leaf)) { throw 'Frontend dependencies are missing. Install the declared frontend dependencies first.' }
    if ($BackendPort -eq $FrontendPort) { throw 'BackendPort and FrontendPort must be different.' }
    Assert-FreePort 'Backend' $BackendPort
    Assert-FreePort 'Frontend' $FrontendPort
    $python = & $resolver -Kind Python
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($python)) { throw 'Python runtime discovery failed.' }
    $node = & $resolver -Kind Node
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($node)) { throw 'Node.js runtime discovery failed.' }
    $startOptions = @{}
    if ($Headless) { $startOptions['NoNewWindow'] = $true } else { $startOptions['WindowStyle'] = 'Hidden' }
    # Process-only overrides: the launcher's printed addresses are authoritative.
    $env:BACKEND_RELOAD = 'false'
    $env:BACKEND_HOST = '127.0.0.1'
    $env:BACKEND_PORT = [string]$BackendPort
    $env:REPONOESIS_STARTUP_ID = $launchId
    $env:REPONOESIS_BACKEND_ORIGIN = $backendUrl
    $env:VITE_API_BASE_URL = $frontendUrl
    $backend = Start-Process -FilePath $python -ArgumentList @('-m', 'app.run_server') -WorkingDirectory $backendRoot -PassThru @startOptions
    $started.Add($backend)
    Write-Output "Backend process started (PID $($backend.Id)); waiting for health check."
    Wait-ForEndpoint 'Backend' "$backendUrl/api/health" $backend
    $frontend = Start-Process -FilePath $node -ArgumentList @('node_modules/vite/bin/vite.js', '--configLoader', 'native', '--host', '127.0.0.1', '--port', [string]$FrontendPort, '--strictPort') -WorkingDirectory $frontendRoot -PassThru @startOptions
    $started.Add($frontend)
    Write-Output "Frontend process started (PID $($frontend.Id)); waiting for HTTP response."
    Wait-ForEndpoint 'Frontend' "$frontendUrl/" $frontend
    if ($backend.HasExited -or $frontend.HasExited) { throw 'A child process exited during startup.' }
    Write-Output 'RepoNoesis backend and frontend are ready.'
    Write-Output "Backend:  $backendUrl (PID $($backend.Id))"
    Write-Output "Frontend: $frontendUrl (PID $($frontend.Id))"
    Write-Output 'To stop, terminate only these printed PIDs and their own children; closing the browser does not stop services.'
    exit 0
} catch {
    Stop-StartedProcesses
    [Console]::Error.WriteLine("RepoNoesis startup failed: $($_.Exception.Message)")
    exit 1
}
