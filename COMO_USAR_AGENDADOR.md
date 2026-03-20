# 🤖 Agendador Automático VS Code

## Como usar:

### 1️⃣ Iniciar o agendador (deixar rodando no VS Code):

```powershell
py agendador_vscode.py
```

Este comando iniciará um processo em background que:
- ⏰ Executa automaticamente `raspagem.py` todos os dias às **20:00**
- 📝 Gera logs em `logs/agendador_YYYYMMDD.log`
- 🔄 Fica rodando continuamente enquanto o VS Code estiver aberto

### 2️⃣ Verificar se está funcionando:

- Você verá a mensagem: `🚀 Agendador iniciado!`
- O terminal ficará aberto mostrando que está ativo
- Logs serão salvos na pasta `logs/`

### 3️⃣ Parar o agendador:

- Pressione `Ctrl + C` no terminal onde está rodando
- Ou feche o terminal

### ⚙️ Configuração:

- **Horário atual**: 20:00 todos os dias
- **Script executado**: `raspagem.py`
- **Logs**: `logs/agendador_YYYYMMDD.log`

### 📋 Requisitos:

- ✅ VS Code aberto
- ✅ Terminal rodando o agendador
- ✅ Python instalado com bibliotecas (schedule, requests, beautifulsoup4, pandas, openpyxl)

### 🔧 Para mudar o horário:

Edite o arquivo `agendador_vscode.py` na linha:
```python
schedule.every().day.at("20:00").do(executar_scraping)
```

Troque "20:00" pelo horário desejado (formato 24h: HH:MM)
