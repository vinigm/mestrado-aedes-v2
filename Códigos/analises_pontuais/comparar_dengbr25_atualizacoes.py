#!/usr/bin/env python3
"""
Análise pontual: Comparar DENGBR25.csv vs DENGBR25_2.csv
para verificar atualizações nos dados de 2025
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def comparar_dengbr25():
    """
    Compara os arquivos DENGBR25.csv e DENGBR25_2.csv para identificar diferenças
    """
    
    print("="*80)
    print(" COMPARAÇÃO: DENGBR25.csv vs DENGBR25_2.csv ".center(80, "="))
    print("="*80)
    
    base_dir = Path(__file__).parents[2]
    pasta_oficiais = base_dir / "Bases de dados" / "bases_oficiais_opendatasus"
    
    arquivo_original = pasta_oficiais / "DENGBR25.csv"
    arquivo_novo = pasta_oficiais / "DENGBR25_2.csv"
    
    print(f"\n📁 Pasta: {pasta_oficiais}")
    print(f"📄 Arquivo original: DENGBR25.csv")
    print(f"📄 Arquivo novo: DENGBR25_2.csv")
    
    # 1. VERIFICAR EXISTÊNCIA DOS ARQUIVOS
    print(f"\n{'-'*60}")
    print("1. VERIFICAÇÃO DE ARQUIVOS")
    print(f"{'-'*60}")
    
    if not arquivo_original.exists():
        print(f"❌ DENGBR25.csv não encontrado!")
        return
    
    if not arquivo_novo.exists():
        print(f"❌ DENGBR25_2.csv não encontrado!")
        return
    
    # Informações básicas dos arquivos
    tamanho_original = arquivo_original.stat().st_size / (1024*1024)  # MB
    tamanho_novo = arquivo_novo.stat().st_size / (1024*1024)  # MB
    
    data_mod_original = datetime.fromtimestamp(arquivo_original.stat().st_mtime)
    data_mod_novo = datetime.fromtimestamp(arquivo_novo.stat().st_mtime)
    
    print(f"\n📊 DENGBR25.csv (original):")
    print(f"   • Tamanho: {tamanho_original:.1f} MB")
    print(f"   • Modificado: {data_mod_original.strftime('%d/%m/%Y %H:%M:%S')}")
    
    print(f"\n📊 DENGBR25_2.csv (novo):")
    print(f"   • Tamanho: {tamanho_novo:.1f} MB")
    print(f"   • Modificado: {data_mod_novo.strftime('%d/%m/%Y %H:%M:%S')}")
    
    diferenca_tamanho = tamanho_novo - tamanho_original
    diferenca_pct = (diferenca_tamanho / tamanho_original) * 100 if tamanho_original > 0 else 0
    
    print(f"\n📈 DIFERENÇA:")
    print(f"   • Tamanho: {diferenca_tamanho:+.1f} MB ({diferenca_pct:+.1f}%)")
    print(f"   • Tempo: {(data_mod_novo - data_mod_original).days} dias de diferença")
    
    # 2. COMPARAÇÃO ESTRUTURAL
    print(f"\n{'-'*60}")
    print("2. COMPARAÇÃO ESTRUTURAL")
    print(f"{'-'*60}")
    
    try:
        print(f"\n🔍 Carregando amostras dos arquivos...")
        
        # Carregar amostras pequenas para comparação estrutural
        print(f"   Carregando 1000 registros de cada arquivo...")
        df_original = pd.read_csv(arquivo_original, nrows=1000, low_memory=False)
        df_novo = pd.read_csv(arquivo_novo, nrows=1000, low_memory=False)
        
        print(f"\n📋 ESTRUTURA ORIGINAL:")
        print(f"   • Registros (amostra): {len(df_original):,}")
        print(f"   • Colunas: {len(df_original.columns)}")
        print(f"   • Período: {df_original['NU_ANO'].min()}-{df_original['NU_ANO'].max()}")
        
        print(f"\n📋 ESTRUTURA NOVA:")
        print(f"   • Registros (amostra): {len(df_novo):,}")
        print(f"   • Colunas: {len(df_novo.columns)}")
        print(f"   • Período: {df_novo['NU_ANO'].min()}-{df_novo['NU_ANO'].max()}")
        
        # Comparar colunas
        colunas_original = set(df_original.columns)
        colunas_novo = set(df_novo.columns)
        
        colunas_comuns = colunas_original & colunas_novo
        colunas_removidas = colunas_original - colunas_novo
        colunas_adicionadas = colunas_novo - colunas_original
        
        print(f"\n🔗 COMPARAÇÃO DE COLUNAS:")
        print(f"   • Colunas em comum: {len(colunas_comuns)}")
        print(f"   • Colunas removidas: {len(colunas_removidas)}")
        print(f"   • Colunas adicionadas: {len(colunas_adicionadas)}")
        
        if colunas_removidas:
            print(f"   ❌ Removidas: {list(colunas_removidas)}")
        if colunas_adicionadas:
            print(f"   ✅ Adicionadas: {list(colunas_adicionadas)}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar arquivos: {e}")
        return
    
    # 3. COMPARAÇÃO TEMPORAL
    print(f"\n{'-'*60}")
    print("3. ANÁLISE TEMPORAL")
    print(f"{'-'*60}")
    
    try:
        print(f"\n📅 DISTRIBUIÇÃO POR SEMANA EPIDEMIOLÓGICA:")
        
        # Analisar distribuição temporal
        if 'SEM_NOT' in df_original.columns and 'SEM_NOT' in df_novo.columns:
            semanas_original = df_original['SEM_NOT'].value_counts().sort_index()
            semanas_novo = df_novo['SEM_NOT'].value_counts().sort_index()
            
            print(f"\n   📊 ARQUIVO ORIGINAL (últimas 10 semanas):")
            for semana, freq in semanas_original.tail(10).items():
                print(f"      • Semana {semana}: {freq:,} casos")
            
            print(f"\n   📊 ARQUIVO NOVO (últimas 10 semanas):")
            for semana, freq in semanas_novo.tail(10).items():
                print(f"      • Semana {semana}: {freq:,} casos")
            
            # Verificar se há semanas novas
            semanas_novas = set(semanas_novo.index) - set(semanas_original.index)
            if semanas_novas:
                print(f"\n   ✅ SEMANAS NOVAS ENCONTRADAS: {sorted(semanas_novas)}")
                for semana in sorted(semanas_novas):
                    print(f"      • Semana {semana}: {semanas_novo[semana]:,} novos casos")
            else:
                print(f"\n   ❓ Nenhuma semana nova identificada na amostra")
    
    except Exception as e:
        print(f"❌ Erro na análise temporal: {e}")
    
    # 4. ANÁLISE ESPECÍFICA PARA PORTO ALEGRE
    print(f"\n{'-'*60}")
    print("4. IMPACTO PARA PORTO ALEGRE")
    print(f"{'-'*60}")
    
    try:
        # Filtrar dados de Porto Alegre (código 431490)
        if 'ID_MUNICIP' in df_original.columns and 'ID_MUNICIP' in df_novo.columns:
            poa_original = df_original[df_original['ID_MUNICIP'] == 431490]
            poa_novo = df_novo[df_novo['ID_MUNICIP'] == 431490]
            
            print(f"\n🏙️  DADOS DE PORTO ALEGRE:")
            print(f"   • Arquivo original: {len(poa_original):,} casos (amostra)")
            print(f"   • Arquivo novo: {len(poa_novo):,} casos (amostra)")
            
            diferenca_poa = len(poa_novo) - len(poa_original)
            print(f"   • Diferença: {diferenca_poa:+,} casos na amostra")
            
            if diferenca_poa > 0:
                print(f"   ✅ Há novos casos de dengue para Porto Alegre!")
            elif diferenca_poa == 0:
                print(f"   ❓ Mesma quantidade na amostra (pode haver diferenças no arquivo completo)")
            else:
                print(f"   ❓ Menos casos na amostra (verificar estrutura)")
                
    except Exception as e:
        print(f"❌ Erro na análise de Porto Alegre: {e}")
    
    # 5. RECOMENDAÇÕES
    print(f"\n{'-'*60}")
    print("5. RECOMENDAÇÕES")
    print(f"{'-'*60}")
    
    print(f"\n💡 BASEADO NA ANÁLISE:")
    
    if abs(diferenca_pct) > 5:  # Diferença significativa de tamanho
        print(f"   ✅ ATUALIZAÇÃO SIGNIFICATIVA detectada ({diferenca_pct:+.1f}%)")
        print(f"   📋 AÇÕES RECOMENDADAS:")
        print(f"      1. Substituir DENGBR25.csv pelo DENGBR25_2.csv")
        print(f"      2. Reprocessar dengue_poa_2025.csv com dados atualizados")
        print(f"      3. Verificar impacto nas análises existentes")
        print(f"      4. Atualizar timestamp dos dados processados")
        
        if diferenca_poa > 0:
            print(f"      5. ⚠️  IMPORTANTE: Novos casos de Porto Alegre detectados!")
            
    elif abs(diferenca_pct) > 1:  # Diferença moderada
        print(f"   ❓ ATUALIZAÇÃO MODERADA detectada ({diferenca_pct:+.1f}%)")
        print(f"   📋 VERIFICAR SE VALE A PENA ATUALIZAR")
        
    else:  # Diferença pequena
        print(f"   ⚪ Diferença pequena ({diferenca_pct:+.1f}%)")
        print(f"   📋 Provavelmente não necessita atualização urgente")
    
    print(f"\n🔄 PRÓXIMOS PASSOS SUGERIDOS:")
    print(f"   1. Decidir se mantém DENGBR25.csv atual ou atualiza")
    print(f"   2. Se atualizar, renomear: DENGBR25_2.csv → DENGBR25.csv")
    print(f"   3. Reprocessar dados de Porto Alegre se necessário")
    print(f"   4. Documentar a atualização no controle de versão")

if __name__ == "__main__":
    comparar_dengbr25()