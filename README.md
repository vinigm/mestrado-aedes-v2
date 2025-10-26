# Projeto Mestrado: Predição de Dengue com Dados de Aedes aegypti

Este repositório contém o projeto de mestrado que integra dados de monitoramento de mosquitos Aedes aegypti coletados via MiAedes com dados oficiais de casos de dengue para desenvolvimento de modelos preditivos em Porto Alegre.

## 🎯 Objetivo

Desenvolver modelos de predição de casos de dengue utilizando dados em tempo real de mosquitos Aedes aegypti coletados através do sistema MiAedes, correlacionando com dados epidemiológicos oficiais do SINAN.

## 📁 Estrutura do Projeto

```
mestrado-aedes-v2/
├── 📄 config.py                    # Configurações do sistema de coleta
├── 🔬 scraping_miaedes.py         # Script principal de coleta MiAedes
├── 📚 README.md                   # Documentação do projeto
├── ⚙️ .gitignore                  # Configurações Git
│
├── 💻 Códigos/                    # Scripts de análise e processamento
│   ├── 📊 analise_comparativa.py        # Análise temporal entre anos (2020-2025)
│   ├── 🔗 analise_mosquitos_vs_casos.py # Correlação mosquitos vs casos dengue
│   └── 📖 README.md                     # Documentação dos scripts
│
├── 🦟 Raspagem/                   # Dados coletados do MiAedes (cronológicos)
│   ├── dados_aedes_20250816_weekid450_62mosquitos.xlsx   # Agosto 2025
│   ├── dados_aedes_20250822_weekid451_42mosquitos.xlsx
│   ├── dados_aedes_20250827_weekid452_92mosquitos.xlsx
│   ├── dados_aedes_20250903_weekid453_45mosquitos.xlsx   # Setembro 2025
│   ├── dados_aedes_20250913_weekid454_74mosquitos.xlsx
│   ├── dados_aedes_20250920_weekid455_158mosquitos.xlsx
│   ├── dados_aedes_20250927_weekid456_193mosquitos.xlsx
│   ├── dados_aedes_20251004_weekid457_157mosquitos.xlsx  # Outubro 2025
│   ├── dados_aedes_20251016_weekid459_306mosquitos.xlsx
│   ├── dados_aedes_20251022_weekid460_203mosquitos.xlsx
│   └── dados_aedes_20251026_weekid461_0mosquitos.xlsx    # Mais recente
│
├── 📈 Bases de dados/             # Dados oficiais SINAN (não versionados)
│   ├── DENGBR20.csv → DENGBR25.csv     # Bases nacionais anuais
│   ├── dengue_poa_2020.csv → 2025.csv  # Dados Porto Alegre processados
│   └── dengue_poa_filtrado.csv         # Dados consolidados
│
└── 🧬 dengue_prediction_fabio_kon/ # Framework base (Fabio Kon et al.)
    ├── main.py                        # Pipeline principal
    ├── mlMethods.py                   # Algoritmos ML implementados  
    ├── measuringResults.py            # Métricas de avaliação
    └── requirements.txt               # Dependências Python
```

## 🚀 Como Usar

### 1. Coleta de Dados MiAedes
```bash
# Coleta dados atuais do sistema MiAedes
python scraping_miaedes.py

# Coleta sem usar cache (força nova requisição)
python scraping_miaedes.py --no-cache

# Modo verboso para debug
python scraping_miaedes.py -v
```

### 2. Análises Implementadas
```bash
# Análise comparativa temporal (2020-2025)
python Códigos/analise_comparativa.py

# Correlação mosquitos vs casos de dengue  
python Códigos/analise_mosquitos_vs_casos.py
```

## 📊 Dados Coletados

### MiAedes (Tempo Real)
- **Fonte**: https://www.miaedes.com.br
- **Frequência**: Semanal (semanas epidemiológicas)
- **Abrangência**: ~910 armadilhas em Porto Alegre
- **Variáveis**: Coordenadas, endereços, contagens por espécie/gênero
- **Período**: Agosto 2025 - presente

### SINAN (Dados Oficiais)
- **Fonte**: Ministério da Saúde / DATASUS
- **Dados**: Casos confirmados de dengue
- **Abrangência**: Porto Alegre, RS, Brasil
- **Período**: 2020-2025

## 🔧 Configuração

### Dependências Principais
```bash
pip install requests beautifulsoup4 pandas openpyxl
```

### Configurações (config.py)
```python
URL_BASE = 'https://www.miaedes.com.br/public-maps/client/72/region/72/weekly'
DEFAULT_OUTPUT = 'Raspagem/dados_aedes_{timestamp}.xlsx'
USAR_CACHE = True
CACHE_DURATION = 3600  # 1 hora
```

## 📈 Metodologia

1. **Coleta Automatizada**: Script scraping_miaedes.py coleta dados semanais
2. **Processamento**: Normalização de coordenadas, endereços e contagens
3. **Correlação**: Análise temporal entre mosquitos capturados e casos reportados
4. **Predição**: Modelos ML usando características do framework Fabio Kon

## 🔗 Nomenclatura de Arquivos

### Padrão MiAedes
`dados_aedes_YYYYMMDD_weekidXXX_XXmosquitos.xlsx`
- **YYYYMMDD**: Data da coleta
- **weekidXXX**: Semana epidemiológica 
- **XXmosquitos**: Total de mosquitos capturados

## 📝 Notas do Desenvolvimento

- **Repositório limpo**: Versão organizada sem duplicatas e arquivos temporários
- **Automação**: Coleta programada e processamento automático
- **Reprodutibilidade**: Configurações versionadas e documentadas
- **Escalabilidade**: Estrutura preparada para expansão temporal e geográfica

## 🎓 Contexto Acadêmico

**Programa**: Mestrado em [Programa]  
**Orientação**: [Nome do Orientador]  
**Linha de Pesquisa**: Predição epidemiológica usando dados de monitoramento entomológico  
**Inovação**: Integração de dados em tempo real de armadilhas com modelos preditivos

---

*Última atualização: Outubro 2025*  
*Dados atualizados até: Semana Epidemiológica 461/2025*