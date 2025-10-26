# RELATÓRIO DE ANÁLISE - PROJETO DENGUE PREDICTION FABIO KON
# ===========================================================

import os
import pandas as pd
from pathlib import Path

def analisar_estrutura_projeto():
    """Analisa a estrutura completa do projeto dengue prediction"""
    
    print("=" * 80)
    print("📋 RELATÓRIO COMPLETO - PROJETO DENGUE PREDICTION FABIO KON")
    print("=" * 80)
    
    # 1. Estrutura de arquivos
    print("\n🏗️  1. ARQUITETURA DO PROJETO")
    print("-" * 50)
    
    arquivos_python = [f for f in os.listdir('.') if f.endswith('.py')]
    
    modulos = {
        'main.py': 'Arquivo principal - orquestra todo o pipeline',
        'mlMethods.py': 'Métodos de Machine Learning',
        'mlSupportMethods.py': 'Métodos de suporte para ML',
        'measuringResults.py': 'Métricas e avaliação de resultados',
        'regression_metrics.py': 'Métricas específicas de regressão',
        'neighbors_methods.py': 'Métodos de análise de vizinhança',
        'apply_grid_search_sarima.py': 'Grid search para modelos SARIMA'
    }
    
    for arquivo in arquivos_python:
        descricao = modulos.get(arquivo, 'Módulo auxiliar')
        print(f"📄 {arquivo:30} - {descricao}")
    
    # 2. Análise dos dados
    print("\n📊 2. ESTRUTURA DE DADOS")
    print("-" * 50)
    
    # Verificar pasta data
    data_path = Path('data')
    if data_path.exists():
        print("✅ Pasta 'data' encontrada")
        
        # Listar arquivos de dados principais
        csv_files = list(data_path.glob('*.csv'))
        excel_files = list(data_path.glob('*.xls*'))
        
        print(f"\n📈 Arquivos CSV encontrados: {len(csv_files)}")
        for arquivo in csv_files[:5]:  # Primeiros 5
            size_mb = arquivo.stat().st_size / (1024*1024)
            print(f"   • {arquivo.name} ({size_mb:.1f} MB)")
        
        if len(csv_files) > 5:
            print(f"   ... e mais {len(csv_files)-5} arquivos")
            
        print(f"\n📊 Arquivos Excel encontrados: {len(excel_files)}")
        for arquivo in excel_files:
            size_mb = arquivo.stat().st_size / (1024*1024)
            print(f"   • {arquivo.name} ({size_mb:.1f} MB)")
    
    # 3. Arquivos de configuração e requirements
    print("\n⚙️  3. CONFIGURAÇÃO E DEPENDÊNCIAS")
    print("-" * 50)
    
    if os.path.exists('requirements.txt'):
        print("✅ requirements.txt encontrado")
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
            print(f"   📦 Total de dependências: {len(lines)}")
            
        # Principais dependências para ML
        principais = ['pandas', 'numpy', 'scikit-learn', 'catboost', 'matplotlib', 'seaborn']
        print("   🔧 Dependências principais:")
        with open('requirements.txt', 'r') as f:
            for line in f:
                for lib in principais:
                    if lib in line.lower():
                        print(f"      • {line.strip()}")
                        break
    
    print("\n🎯 4. FOCO DO PROJETO ORIGINAL")
    print("-" * 50)
    print("• 🏙️  Local: Rio de Janeiro (bairros)")
    print("• 📅 Período: 2011-2020")
    print("• 🎯 Objetivo: Predição de casos de dengue")
    print("• ⏰ Horizontes: 1, 2, 3 meses à frente")
    print("• 🤖 Modelo principal: CatBoost")
    
    return True

if __name__ == "__main__":
    analisar_estrutura_projeto()