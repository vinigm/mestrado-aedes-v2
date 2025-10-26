"""Configurações para o script de coleta de dados do MiAedes."""

# Configurações da URL e headers
URL_BASE = 'https://www.miaedes.com.br/public-maps/client/72/region/72/weekly'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Configurações de cache
USAR_CACHE = True
CACHE_DURATION = 3600  # segundos (1 hora)
CACHE_FILE = 'dados_cache.json'

# Configurações de output
DEFAULT_OUTPUT = 'Raspagem/dados_aedes_{timestamp}.xlsx'

# Configurações de timeout
REQUEST_TIMEOUT = 30  # segundos
