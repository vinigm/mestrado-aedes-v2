import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def carregar_dados_raspagem(pasta_raspagem):
    # Pega o arquivo mais recente da pasta Raspagem
    arquivos = list(pasta_raspagem.glob('dados_aedes_*.xlsx'))
    arquivo_mais_recente = max(arquivos, key=lambda x: x.stat().st_mtime)
    
    print(f"Carregando dados de raspagem do arquivo: {arquivo_mais_recente.name}")
    
    # Carrega o arquivo Excel
    df_aedes = pd.read_excel(arquivo_mais_recente)
    
    print("\nColunas disponíveis:")
    print(df_aedes.columns.tolist())
    print("\nPrimeiros registros:")
    print(df_aedes[['week', 'aedes_aegypti_femea', 'aedes_aegypti_macho']].head())
    
    # Soma os mosquitos Aedes aegypti (machos e fêmeas)
    df_aedes['total_aedes'] = (
        df_aedes['aedes_aegypti_femea'].fillna(0) + 
        df_aedes['aedes_aegypti_macho'].fillna(0)
    )
    
    # Formata a semana para o padrão "SS/AAAA"
    df_aedes['week_formatted'] = df_aedes['week'].apply(
        lambda x: f"{x.split('/')[0].zfill(2)}/{2025}"
    )
    
    # Agrupa por semana e soma os mosquitos
    df_aedes_semanal = df_aedes.groupby('week_formatted')['total_aedes'].sum().reset_index()
    df_aedes_semanal = df_aedes_semanal.rename(columns={
        'week_formatted': 'week',
        'total_aedes': 'mosquitos_capturados'
    })
    
    print("\nDados agrupados por semana:")
    print(df_aedes_semanal.head())
    
    return df_aedes_semanal

def carregar_dados_dengue(pasta_bases):
    # Lista todos os arquivos dengue_poa_*.csv
    arquivos = list(pasta_bases.glob('dengue_poa_*.csv'))
    
    # Lista para armazenar os dataframes
    dfs = []
    
    # Carrega cada arquivo
    for arquivo in arquivos:
        if arquivo.name == 'dengue_poa_filtrado.csv':
            continue
            
        print(f"Carregando dados de dengue do arquivo: {arquivo.name}")
        df = pd.read_csv(arquivo)
        
        # Verifica se as colunas necessárias existem
        if 'week' not in df.columns:
            print(f"Arquivo {arquivo.name} não tem a coluna 'week'. Pulando...")
            continue
            
        # Conta os casos por semana
        casos_semana = df.groupby('week').size().reset_index(name='casos_dengue')
        dfs.append(casos_semana)
    
    # Combina todos os dataframes
    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        # Agrupa novamente para somar casos de semanas que podem aparecer em mais de um arquivo
        df_final = df_final.groupby('week')['casos_dengue'].sum().reset_index()
        return df_final
    else:
        return None

def main():
    # Define os caminhos das pastas
    base_dir = Path(__file__).parents[1]
    pasta_raspagem = base_dir / "Raspagem"
    pasta_bases = base_dir / "Bases de dados"
    
    # Carrega os dados
    df_aedes = carregar_dados_raspagem(pasta_raspagem)
    df_dengue = carregar_dados_dengue(pasta_bases)
    
    if df_dengue is None:
        print("Não foi possível carregar os dados de dengue!")
        return
    
    # Faz o merge dos dados
    df_final = pd.merge(df_aedes, df_dengue, on='week', how='outer')
    
    # Preenche valores nulos com 0
    df_final = df_final.fillna(0)
    
    # Ordena por semana
    df_final['ano'] = df_final['week'].str.split('/').str[1].astype(int)
    df_final['semana'] = df_final['week'].str.split('/').str[0].astype(int)
    df_final = df_final.sort_values(['ano', 'semana'])
    
    # Adiciona colunas para análise
    df_final['casos_por_mosquito'] = df_final.apply(
        lambda row: row['casos_dengue'] / row['mosquitos_capturados'] 
        if row['mosquitos_capturados'] > 0 else 0, axis=1
    )
    
    # Remove colunas temporárias de ordenação
    df_final = df_final.drop(['ano', 'semana'], axis=1)
    
    # Salva o resultado
    arquivo_saida = base_dir / "Bases de dados" / "analise_mosquitos_vs_casos.csv"
    df_final.to_csv(arquivo_saida, index=False)
    
    print("\nResumo dos dados:")
    print("-" * 50)
    print(f"Total de semanas analisadas: {len(df_final)}")
    print(f"Total de mosquitos capturados: {df_final['mosquitos_capturados'].sum():.0f}")
    print(f"Total de casos de dengue: {df_final['casos_dengue'].sum():.0f}")
    print("\nMétricas por semana:")
    print(f"Média de mosquitos capturados: {df_final['mosquitos_capturados'].mean():.2f}")
    print(f"Média de casos de dengue: {df_final['casos_dengue'].mean():.2f}")
    print(f"Média de casos por mosquito: {df_final['casos_por_mosquito'].mean():.4f}")
    print(f"\nPrimeiras linhas do arquivo gerado:")
    print(df_final.head())
    print(f"\nArquivo salvo em: {arquivo_saida}")

if __name__ == "__main__":
    main()
