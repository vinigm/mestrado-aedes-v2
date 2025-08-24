# Análise de Dados do Aedes aegypti

Este repositório contém scripts e ferramentas para coleta, processamento e análise de dados relacionados ao mosquito Aedes aegypti.

## Estrutura do Projeto

```
.
├── Códigos/                  # Scripts Python para análise e processamento
│   ├── analise_comparativa.py      # Análise comparativa entre períodos
│   ├── config.py                   # Configurações gerais do projeto
│   ├── script.py                   # Script original de coleta
│   ├── script_new.py              # Nova versão do script de coleta
│   ├── teste_mexer_base_25.py     # Testes com base de dados 2025
│   ├── verificar_cidades_rs_com_casos  # Verificação de casos por cidade
│   └── verificar_estado.py        # Análise por estado
│
└── Raspagem/                # Dados coletados do monitoramento
    ├── dados_aedes_20250816_194136.xlsx  # Dados coletados em 16/08/2025
    └── dados_aedes_20250822_095411.xlsx  # Dados coletados em 22/08/2025
```

## Componentes Principais

### Scripts de Análise (`Códigos/`)

- **analise_comparativa.py**: Realiza análises comparativas entre diferentes períodos
- **config.py**: Contém configurações globais do projeto
- **script.py** e **script_new.py**: Scripts para coleta de dados, sendo o `script_new.py` uma versão melhorada
- **verificar_cidades_rs_com_casos** e **verificar_estado.py**: Scripts para análise geográfica dos casos

### Dados Coletados (`Raspagem/`)

Contém os arquivos Excel com os dados coletados do monitoramento do Aedes aegypti. Os arquivos são nomeados com a data e hora da coleta para rastreabilidade.

## Observações

- A pasta "Bases de dados" contém dados locais e não é versionada no repositório
- Os dados históricos na pasta "Raspagem" são mantidos para referência
- Arquivos temporários e de cache são ignorados no versionamento