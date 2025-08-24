# Mestrado Aedes

Script de coleta e processamento de dados do MiAedes para análise de infestação de mosquitos.

## Funcionalidades

- Coleta automática de dados do MiAedes
- Processamento e limpeza dos dados
- Exportação para Excel
- Sistema de cache para otimizar requisições
- Logging detalhado
- Interface de linha de comando (CLI)

## Requisitos

```bash
pip install requests beautifulsoup4 pandas openpyxl
```

## Uso

Uso básico:
```bash
python script_new.py
```

Opções disponíveis:
```bash
python script_new.py -o "saida.xlsx"  # Define arquivo de saída
python script_new.py -v               # Modo verbose
python script_new.py --no-cache       # Desativa cache
```

## Estrutura do Projeto

- `script_new.py`: Script principal com código refatorado e melhorias
- `config.py`: Arquivo de configurações
- `script.py`: Versão original do script (mantido para referência)

## Dados Coletados

O script coleta e processa:
- Contagens de Aedes aegypti (macho/fêmea)
- Contagens de Aedes albopictus (macho/fêmea)
- Contagens de Culex sp (macho/fêmea)
- Coordenadas geográficas
- Dados de endereço
- Total de mosquitos por ponto
