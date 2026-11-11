param(
    [switch]$DisableHyDE,
    [switch]$DisableRerank,
    [ValidateRange(1, 65535)][int]$BackendPort = 8000,
    [ValidateRange(1, 65535)][int]$FrontendPort = 5173,
    [switch]$Headless
)
$ErrorActionPreference = 'Stop'
# Overrides belong only to this launch and its children. No .env writes.
$env:RETRIEVAL_HYDE_ENABLED = if ($DisableHyDE) { 'false' } else { 'true' }
$env:RETRIEVAL_RERANK_ENABLED = if ($DisableRerank) { 'false' } else { 'true' }
$env:RETRIEVAL_ENHANCEMENT_SECONDS = '15'
& (Join-Path $PSScriptRoot 'start_local.ps1') -BackendPort $BackendPort -FrontendPort $FrontendPort -Headless:$Headless
