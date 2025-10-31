# Configurar execução automática do scraping MiAedes
# Execute como Administrador

# Verificar administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "ERRO: Execute como Administrador!" -ForegroundColor Red
    Read-Host "Pressione Enter para sair"
    exit 1
}

# Configurações
$ScriptPath = "G:\Meu Drive\Mestrado\Mestrado mesmo\ScriptScrapping"
$ScriptFile = "executar_scraping_automatico.py"
$TaskName = "ScrapingMiAedes"
$ExecutionTime = "20:00"

# Verificar Python (Jeito robusto)
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { $null }

if (-not $pythonCmd) {
    Write-Host "ERRO: Python (python/py) não encontrado no PATH!" -ForegroundColor Red
    Read-Host "Pressione Enter para sair"
    exit 1
}

$fullScriptPath = Join-Path $ScriptPath $ScriptFile

# Remover tarefa existente e criar nova
schtasks /delete /tn $TaskName /f 2>$null | Out-Null

# Criar tarefa com diretório de trabalho
$taskAction = "$pythonCmd `"$fullScriptPath`""
$result = cmd /c "schtasks /Create /TN `"$TaskName`" /SC DAILY /ST $ExecutionTime /TR `"$taskAction`" /F" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Tarefa configurada com sucesso!" -ForegroundColor Green
    Write-Host "• Tarefa: $TaskName"
    Write-Host "• Hora: $ExecutionTime"
    Write-Host "• Ação: $taskAction"
    Write-Host "• Pasta de Trabalho: Gerenciado pelo Python"
} else {
    Write-Host "Erro ao criar tarefa:" -ForegroundColor Red
    Write-Host $result -ForegroundColor Red
    exit 1
}

Read-Host "Pressione Enter para finalizar"