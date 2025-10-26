#!/usr/bin/env python3
"""
Análise pontual: Verificar se os arquivos dengue_poa_202* são processamentos 
das bases oficiais DENGBR* filtrados para Porto Alegre.

Criado para esclarecer a origem e processamento dos dados locais.
"""

import pandas as pd
import os
from pathlib import Path

def analisar_origem_dados():
    """
    Compara estrutura e conteúdo entre:
    - Bases oficiais: DENGBR*.csv (pasta bases_oficiais_opendatasus)
    - Dados locais: dengue_poa_*.csv (pasta Bases de dados)
    """
    
    print("="*80)
    print(" ANÁLISE: ORIGEM DOS ARQUIVOS dengue_poa_202* ".center(80, "="))
    print("="*80)
    
    # Caminhos das pastas
    base_dir = Path(__file__).parents[2]
    pasta_oficiais = base_dir / "Bases de dados" / "bases_oficiais_opendatasus"
    pasta_locais = base_dir / "Bases de dados"
    
    print(f"\n📁 Pasta bases oficiais: {pasta_oficiais}")
    print(f"📁 Pasta dados locais: {pasta_locais}")
    
    # 1. VERIFICAR ARQUIVOS DISPONÍVEIS
    print(f"\n{'-'*60}")
    print("1. ARQUIVOS ENCONTRADOS")
    print(f"{'-'*60}")
    
    # Arquivos oficiais DENGBR
    if pasta_oficiais.exists():
        arquivos_oficiais = list(pasta_oficiais.glob("DENGBR*.csv"))
        print(f"\n📋 Bases oficiais encontradas ({len(arquivos_oficiais)}):")
        for arq in sorted(arquivos_oficiais):
            tamanho = arq.stat().st_size / (1024*1024)  # MB
            print(f"   • {arq.name} ({tamanho:.1f} MB)")
    else:
        print(f"\n❌ Pasta de bases oficiais não encontrada!")
        return
    
    # Arquivos locais dengue_poa
    arquivos_poa = list(pasta_locais.glob("dengue_poa_202*.csv"))
    print(f"\n📋 Arquivos dengue_poa encontrados ({len(arquivos_poa)}):")
    for arq in sorted(arquivos_poa):
        tamanho = arq.stat().st_size / (1024*1024)  # MB
        print(f"   • {arq.name} ({tamanho:.1f} MB)")
    
    # 2. COMPARAR ESTRUTURA DE DADOS
    print(f"\n{'-'*60}")
    print("2. ANÁLISE ESTRUTURAL DOS DADOS")
    print(f"{'-'*60}")
    
    # Analisar um arquivo oficial (exemplo: 2024)
    arquivo_oficial_2024 = pasta_oficiais / "DENGBR24.csv"
    arquivo_poa_2024 = pasta_locais / "dengue_poa_2024.csv"
    
    if arquivo_oficial_2024.exists() and arquivo_poa_2024.exists():
        print(f"\n🔍 Comparando: DENGBR24.csv vs dengue_poa_2024.csv")
        
        # Carregar amostras dos dados
        print("   Carregando amostras (primeiras 1000 linhas)...")
        df_oficial = pd.read_csv(arquivo_oficial_2024, nrows=1000)
        df_poa = pd.read_csv(arquivo_poa_2024, nrows=1000)
        
        print(f"\n📊 DENGBR24.csv:")
        print(f"   • Total colunas: {len(df_oficial.columns)}")
        print(f"   • Primeiras colunas: {list(df_oficial.columns[:5])}")
        print(f"   • Amostra registros: {len(df_oficial)}")
        
        print(f"\n📊 dengue_poa_2024.csv:")
        print(f"   • Total colunas: {len(df_poa.columns)}")
        print(f"   • Primeiras colunas: {list(df_poa.columns[:5])}")
        print(f"   • Amostra registros: {len(df_poa)}")
        
        # Verificar se as colunas são as mesmas
        colunas_comuns = set(df_oficial.columns) & set(df_poa.columns)
        colunas_diferentes = set(df_oficial.columns) ^ set(df_poa.columns)
        
        print(f"\n🔗 Colunas em comum: {len(colunas_comuns)}")
        print(f"❓ Colunas diferentes: {len(colunas_diferentes)}")
        
        if colunas_diferentes:
            print("   Diferenças encontradas:")
            only_oficial = set(df_oficial.columns) - set(df_poa.columns)
            only_poa = set(df_poa.columns) - set(df_oficial.columns)
            
            if only_oficial:
                print(f"     • Apenas em DENGBR24: {list(only_oficial)[:3]}...")
            if only_poa:
                print(f"     • Apenas em dengue_poa: {list(only_poa)[:3]}...")
    
    # 3. VERIFICAR FILTRO POR PORTO ALEGRE
    print(f"\n{'-'*60}")
    print("3. VERIFICAÇÃO DE FILTRO POR PORTO ALEGRE")
    print(f"{'-'*60}")
    
    if arquivo_oficial_2024.exists():
        print(f"\n🔍 Verificando códigos de município em DENGBR24...")
        
        # Verificar se existe coluna de município
        df_sample = pd.read_csv(arquivo_oficial_2024, nrows=5000)
        colunas_municipio = [col for col in df_sample.columns 
                           if 'MUNICIP' in col.upper() or 'CIDADE' in col.upper()]
        
        print(f"   Colunas relacionadas a município: {colunas_municipio}")
        
        if colunas_municipio:
            col_municipio = colunas_municipio[0]
            municipios_unicos = df_sample[col_municipio].value_counts().head(10)
            print(f"\n   Top 10 códigos de município (amostra):")
            for codigo, freq in municipios_unicos.items():
                print(f"     • {codigo}: {freq} casos")
                
            # Códigos conhecidos de Porto Alegre
            codigos_poa = [4314902, 431490, 43149]  # Variações possíveis
            poa_encontrado = False
            for codigo in codigos_poa:
                if codigo in municipios_unicos.index:
                    print(f"\n✅ Código de Porto Alegre encontrado: {codigo}")
                    print(f"   Frequência na amostra: {municipios_unicos[codigo]} casos")
                    poa_encontrado = True
                    break
            
            if not poa_encontrado:
                print(f"\n❓ Códigos conhecidos de POA não encontrados na amostra")
                print(f"   Códigos testados: {codigos_poa}")
    
    # 4. CONCLUSÕES PRELIMINARES
    print(f"\n{'-'*60}")
    print("4. CONCLUSÕES PRELIMINARES")
    print(f"{'-'*60}")
    
    print(f"\n🎯 HIPÓTESE TESTADA:")
    print(f"   Os arquivos dengue_poa_202*.csv são processamentos dos")
    print(f"   arquivos DENGBR*.csv filtrados para Porto Alegre")
    
    print(f"\n💡 EVIDÊNCIAS ENCONTRADAS:")
    if 'df_oficial' in locals() and 'df_poa' in locals():
        if len(colunas_comuns) > len(colunas_diferentes):
            print(f"   ✅ Estruturas similares (muitas colunas em comum)")
        else:
            print(f"   ❓ Estruturas diferentes (poucos campos em comum)")
            
        if len(df_poa) < len(df_oficial):
            print(f"   ✅ Arquivo POA menor (indica possível filtro)")
        else:
            print(f"   ❓ Arquivo POA similar/maior (filtro questionável)")
    
    print(f"\n📋 PRÓXIMOS PASSOS SUGERIDOS:")
    print(f"   1. Verificar processamento completo dos dados")
    print(f"   2. Confirmar códigos de município de Porto Alegre")
    print(f"   3. Comparar totais de registros por ano")
    print(f"   4. Analisar scripts de processamento existentes")

if __name__ == "__main__":
    analisar_origem_dados()