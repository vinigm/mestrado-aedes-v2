# Script para analisar dados de dengue em Porto Alegre das bases de 2024 e 2025

import pandas as pd
import os
from pathlib import Path

# Configurações
BASE_DIR = Path(r'C:\Users\Vinicius Mello\OneDrive\Mestrado\Mestrado mesmo\ScriptScrapping')
ARQUIVO_2024 = BASE_DIR / 'Bases de dados' / 'DENGBR24.csv'
ARQUIVO_2025 = BASE_DIR / 'Bases de dados' / 'DENGBR25.csv'

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
# O arquivo é grande, isso pode demorar e consumir memória
df_dengue = pd.read_csv(r'C:\Users\Vinicius Mello\OneDrive\Mestrado\Mestrado mesmo\ScriptScrapping\Bases de dados\DENGBR25.csv', sep=',', encoding='utf-8')

# Diagnóstico dos tipos de dados
print("\n=== DIAGNÓSTICO DOS TIPOS DE DADOS ===")
print("\n1. Tipos das colunas principais:")
print(f"ID_MUNICIP: {df_dengue['ID_MUNICIP'].dtype}")
print(f"CLASSI_FIN: {df_dengue['CLASSI_FIN'].dtype}")

# Exemplo dos primeiros valores
print("\n2. Primeiros valores das colunas (para inspeção):")
print("\nID_MUNICIP primeiros 5 valores:")
print(df_dengue['ID_MUNICIP'].head())
print("\nCLASSI_FIN primeiros 5 valores:")
print(df_dengue['CLASSI_FIN'].head())

# Verificar total de notificações de Porto Alegre (sem filtro de confirmação)
codigo_poa = 4314902
total_poa = df_dengue[df_dengue['ID_MUNICIP'] == codigo_poa].shape[0]
print(f"\n3. Total de notificações em Porto Alegre (sem filtro): {total_poa}")

# Se os tipos não forem números, converter
if df_dengue['ID_MUNICIP'].dtype == 'object':
    print("\nConvertendo ID_MUNICIP para número...")
    df_dengue['ID_MUNICIP'] = pd.to_numeric(df_dengue['ID_MUNICIP'], errors='coerce')

if df_dengue['CLASSI_FIN'].dtype == 'object':
    print("Convertendo CLASSI_FIN para número...")
    df_dengue['CLASSI_FIN'] = pd.to_numeric(df_dengue['CLASSI_FIN'], errors='coerce')

# Define os códigos de classificação final para dengue confirmada
casos_confirmados = [10, 11, 12]

# Aplica os filtros após a conversão
df_dengue_poa = df_dengue[
    (df_dengue['ID_MUNICIP'] == codigo_poa) &
    (df_dengue['CLASSI_FIN'].isin(casos_confirmados))
].copy()

# Verificar quantos casos foram encontrados após a conversão
print(f"\n4. Casos confirmados após conversão de tipo: {len(df_dengue_poa)}")

# Se ainda não encontrou casos, vamos verificar os valores únicos
if len(df_dengue_poa) == 0:
    print("\n=== ANÁLISE ADICIONAL ===")
    print("\nValores únicos em CLASSI_FIN:")
    print(df_dengue['CLASSI_FIN'].value_counts().head())
    print("\nPrimeiras linhas de Porto Alegre:")
    print(df_dengue[df_dengue['ID_MUNICIP'] == codigo_poa].head())

# Agora df_dengue_poa contém apenas os dados que você precisa
print(f"\nForam encontrados {len(df_dengue_poa)} casos confirmados de dengue em Porto Alegre.")

# Análise detalhada dos dados
print("\n" + "="*50)
print("RESUMO DOS DADOS DE DENGUE EM PORTO ALEGRE")
print("="*50)

# Total de casos por ano
print("\n1. Distribuição temporal:")
casos_por_ano = df_dengue_poa['DT_NOTIFIC'].str[:4].value_counts().sort_index()
print("\nCasos por ano:")
print(casos_por_ano)

# Distribuição por sexo (se disponível)
if 'CS_SEXO' in df_dengue_poa.columns:
    print("\n2. Distribuição por sexo:")
    sexo_counts = df_dengue_poa['CS_SEXO'].value_counts()
    print(sexo_counts)

# Distribuição por faixa etária (se disponível)
if 'NU_IDADE_N' in df_dengue_poa.columns:
    print("\n3. Estatísticas de idade:")
    print(df_dengue_poa['NU_IDADE_N'].describe())

# Evolução dos casos (se disponível)
if 'EVOLUCAO' in df_dengue_poa.columns:
    print("\n4. Evolução dos casos:")
    evolucao_counts = df_dengue_poa['EVOLUCAO'].value_counts()
    print(evolucao_counts)

# Casos por mês (análise sazonal)
print("\n5. Distribuição mensal:")
df_dengue_poa['MES'] = df_dengue_poa['DT_NOTIFIC'].str[5:7]
casos_por_mes = df_dengue_poa['MES'].value_counts().sort_index()
print("\nCasos por mês (todos os anos):")
print(casos_por_mes)

# Estatísticas gerais
print("\n" + "="*50)
print("ESTATÍSTICAS GERAIS:")
print(f"Total de casos confirmados: {len(df_dengue_poa)}")
print(f"Período dos dados: de {df_dengue_poa['DT_NOTIFIC'].min()} até {df_dengue_poa['DT_NOTIFIC'].max()}")
print("="*50)

# Salvar os dados filtrados
output_file = BASE_DIR / 'Bases de dados' / 'dengue_poa_filtrado.csv'
df_dengue_poa.to_csv(output_file, index=False)
print(f"\nDados salvos em: {output_file}")