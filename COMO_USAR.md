# 🦟 Sistema de Scraping MiAedes

Sistema automatizado para coleta de dados do portal MiAedes.

## 📁 Arquivos

- **`raspagem.py`** - Script principal de scraping
- **`executar_scraping_automatico.py`** - Executor automático com logs
- **`configurar_tarefa.ps1`** - Configura execução automática (Windows)
- **`config.py`** - Configurações

## 🚀 Uso

### Manual
```bash
py raspagem.py
```

### Automático
```bash
# Como Administrador
.\configurar_tarefa.ps1
```

### Gerenciar
```bash
# Status
schtasks /query /tn "ScrapingMiAedes"

# Executar agora
schtasks /run /tn "ScrapingMiAedes"

# Remover
schtasks /delete /tn "ScrapingMiAedes" /f
```

## 📊 Saídas

- **Dados:** `Raspagem/*.xlsx`
- **Logs:** `logs/scraping_YYYYMMDD.log`