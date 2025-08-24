# Script para analisar dados de dengue em Porto Alegre das bases de 2024 e 2025

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Configurações
# Obtém o diretório do script atual
SCRIPT_DIR = Path(__file__).parent
# Diretório raiz é um nível acima
BASE_DIR = SCRIPT_DIR.parent

# Lista de anos para análise
ANOS = range(2020, 2026)  # 2020 a 2025
ARQUIVOS = {
    ano: BASE_DIR / 'Bases de dados' / f'DENGBR{str(ano)[2:]}.csv'
    for ano in ANOS
}

# Criar diretório de saída se não existir
OUTPUT_DIR = BASE_DIR / 'Bases de dados'
OUTPUT_DIR.mkdir(exist_ok=True)

def analisar_base(arquivo, ano):
    print(f"\n{'='*60}")
    print(f" ANÁLISE DA BASE DE DADOS DE {ano} ".center(60, '='))
    print('='*60)

    if not arquivo.exists():
        print(f"Arquivo de {ano} não encontrado: {arquivo}")
        return None

    # Carrega o arquivo de dados de dengue
    print(f"\nCarregando dados de {ano}...")
    df = pd.read_csv(arquivo, sep=',', encoding='utf-8')

    # Diagnóstico dos tipos de dados e conversão se necessário
    if df['ID_MUNICIP'].dtype == 'object':
        df['ID_MUNICIP'] = pd.to_numeric(df['ID_MUNICIP'], errors='coerce')
    if df['CLASSI_FIN'].dtype == 'object':
        df['CLASSI_FIN'] = pd.to_numeric(df['CLASSI_FIN'], errors='coerce')

    # Análise dos dados
    # NOTA: Código original do IBGE para Porto Alegre é 4314902,
    # mas nos arquivos está como 431490 (truncado)
    codigo_poa = 431490
    casos_confirmados = [10, 11, 12]

    # Verificar códigos únicos encontrados (diagnóstico)
    codigos_unicos = df['ID_MUNICIP'].unique()
    print("\nDiagnóstico de códigos municipais:")
    print(f"Número de municípios únicos: {len(codigos_unicos)}")
    print("Amostra dos 10 primeiros códigos encontrados:", sorted(codigos_unicos)[:10])
    
    # Total de notificações de Porto Alegre
    total_poa = df[df['ID_MUNICIP'] == codigo_poa].shape[0]
    
    # Casos confirmados
    df_poa = df[
        (df['ID_MUNICIP'] == codigo_poa) &
        (df['CLASSI_FIN'].isin(casos_confirmados))
    ].copy()

    # Resultados
    print(f"\nResultados para {ano}:")
    print(f"Total de notificações em Porto Alegre: {total_poa}")
    print(f"Casos confirmados de dengue: {len(df_poa)}")

    if len(df_poa) > 0:
        # Análise temporal
        df_poa['MES'] = pd.to_datetime(df_poa['DT_NOTIFIC']).dt.month
        casos_por_mes = df_poa['MES'].value_counts().sort_index()
        
        print("\nDistribuição mensal dos casos confirmados:")
        print(casos_por_mes)

        # Distribuição por classificação
        print("\nDistribuição por classificação final:")
        print(df_poa['CLASSI_FIN'].value_counts())

    return df_poa

# Analisar todas as bases
print("\nIniciando análise comparativa das bases de 2020 a 2025...")

# Dicionário para armazenar os resultados
resultados = {}

# Analisar cada ano
for ano in ANOS:
    df = analisar_base(ARQUIVOS[ano], ano)
    if df is not None:
        resultados[ano] = df

# Comparação
print("\n" + "="*60)
print(" COMPARAÇÃO ENTRE OS ANOS ".center(60, '='))
print("="*60)

# Total de casos por ano
print("\nTotal de casos confirmados por ano:")
for ano, df in resultados.items():
    print(f"{ano}: {len(df)} casos")
    
# Períodos de dados
print("\nPeríodos de dados por ano:")
for ano, df in resultados.items():
    if len(df) > 0:
        print(f"{ano}: de {df['DT_NOTIFIC'].min()} até {df['DT_NOTIFIC'].max()}")

# Análise da evolução anual
print("\nEvolução mensal dos casos por ano:")
evolucao_mensal = {}
for ano, df in resultados.items():
    if len(df) > 0:
        evolucao_mensal[ano] = df['MES'].value_counts().sort_index()

# Mostrar evolução mensal lado a lado
meses = range(1, 13)
print("\nCasos por mês em cada ano:")
print("Mês", end="")
for ano in ANOS:
    print(f"  {ano:>6}", end="")
print()
print("-" * 50)

for mes in meses:
    print(f"{mes:2d}", end="")
    for ano in ANOS:
        if ano in evolucao_mensal and mes in evolucao_mensal[ano]:
            print(f"  {evolucao_mensal[ano][mes]:>6}", end="")
        else:
            print("       ", end="")
    print()

# Salvar dados filtrados
for ano, df in resultados.items():
    if len(df) > 0:
        output_file = BASE_DIR / 'Bases de dados' / f'dengue_poa_{ano}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nDados de {ano} salvos em: {output_file}")
