import pandas as pd
import os
from pathlib import Path

def formatar_semana(sem_not):
    try:
        if pd.isna(sem_not):
            return None
        # Converte para string caso não seja
        sem_not = str(sem_not)
        # Verifica se o formato é válido (6 dígitos)
        if not sem_not.isdigit() or len(sem_not) != 6:
            return None
        # Extrai o ano (primeiros 4 dígitos) e a semana (2 últimos dígitos)
        ano = sem_not[:4]
        semana = sem_not[-2:]
        # Retorna no formato "SS/AAAA"
        return f"{semana}/{ano}"
    except:
        return None

def processar_arquivo(arquivo):
    print(f"Processando {arquivo}...")
    try:
        # Lê o arquivo CSV com low_memory=False para evitar warnings de tipos mistos
        df = pd.read_csv(arquivo, low_memory=False)
        
        # Verifica se a coluna SEM_NOT existe
        if 'SEM_NOT' not in df.columns:
            print(f"Arquivo {arquivo} não contém a coluna SEM_NOT")
            return
        
        # Converte SEM_NOT para string para garantir consistência
        df['SEM_NOT'] = df['SEM_NOT'].astype(str)
        
        # Valida o formato da coluna SEM_NOT
        formato_invalido = df['SEM_NOT'].apply(lambda x: len(x) != 6 or not x.isdigit())
        if formato_invalido.any():
            print(f"Aviso: Encontrados valores inválidos em SEM_NOT no arquivo {arquivo}")
            print("Exemplos de valores inválidos:")
            print(df.loc[formato_invalido, 'SEM_NOT'].head())
        
        # Cria a nova coluna 'week' aplicando a função formatar_semana
        df['week'] = df['SEM_NOT'].apply(formatar_semana)
        
        # Conta registros com semana inválida
        registros_invalidos = df['week'].isna().sum()
        if registros_invalidos > 0:
            print(f"\nEncontrados {registros_invalidos} registros com semana epidemiológica inválida")
            print("Removendo registros inválidos...")
            df = df.dropna(subset=['week'])
        
        # Salva o arquivo com as mudanças
        df.to_csv(arquivo, index=False)
        print(f"Arquivo {arquivo} processado com sucesso!")
        
        # Mostra algumas linhas de exemplo do resultado
        print("\nExemplo de algumas linhas processadas:")
        print(df[['SEM_NOT', 'week']].head())
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"Erro ao processar o arquivo {arquivo}: {str(e)}")

def main():
    # Diretório onde estão os arquivos
    base_dir = Path(__file__).parents[1] / "Bases de dados"
    
    # Processa todos os arquivos dengue_poa_*.csv
    for arquivo in base_dir.glob("dengue_poa_*.csv"):
        processar_arquivo(arquivo)

if __name__ == "__main__":
    main()
