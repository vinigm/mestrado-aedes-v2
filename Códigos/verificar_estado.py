import pandas as pd
from pathlib import Path

# Configure o seu diretório base
BASE_DIR = Path(r'C:\Users\Vinicius Mello\OneDrive\Mestrado\Mestrado mesmo\ScriptScrapping')
ARQUIVO_2024 = BASE_DIR / 'Bases de dados' / 'DENGBR24.csv'
ARQUIVO_2025 = BASE_DIR / 'Bases de dados' / 'DENGBR25.csv'

print("--- Verificando conteúdo do arquivo DENGBR24.csv ---")
try:
    if ARQUIVO_2024.exists():
        # Lê apenas a coluna do estado para ser mais rápido
        df24 = pd.read_csv(ARQUIVO_2024, usecols=['SG_UF_NOT'], low_memory=False)
        estados24 = df24['SG_UF_NOT'].unique()
        print(f"Estados encontrados em 2024: {estados24}")
        if 43 in estados24:
            print(">>> SUCESSO: Rio Grande do Sul (código 43) está no arquivo de 2024.")
        else:
            print(">>> ATENÇÃO: Rio Grande do Sul (código 43) NÃO foi encontrado no arquivo de 2024.")
    else:
        print("Arquivo de 2024 não encontrado.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo de 2024: {e}")


print("\n--- Verificando conteúdo do arquivo DENGBR25.csv ---")
try:
    if ARQUIVO_2025.exists():
        # Lê apenas a coluna do estado para ser mais rápido
        df25 = pd.read_csv(ARQUIVO_2025, usecols=['SG_UF_NOT'], low_memory=False)
        estados25 = df25['SG_UF_NOT'].unique()
        print(f"Estados encontrados em 2025: {estados25}")
        if 43 in estados25:
            print(">>> SUCESSO: Rio Grande do Sul (código 43) está no arquivo de 2025.")
        else:
            print(">>> ATENÇÃO: Rio Grande do Sul (código 43) NÃO foi encontrado no arquivo de 2025.")
    else:
        print("Arquivo de 2025 não encontrado.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo de 2025: {e}")