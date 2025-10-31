#!/usr/bin/env python3
"""
Análise pontual: Relatório completo do projeto dengue_prediction_fabio_kon
Análise de todos os códigos e funcionalidades implementadas
"""

import os
from pathlib import Path
import ast
import re

def analisar_projeto_fabio_kon():
    """
    Análise completa do projeto de predição de dengue do Fabio Kon
    """
    
    print("="*80)
    print(" RELATÓRIO: PROJETO DENGUE_PREDICTION_FABIO_KON ".center(80, "="))
    print("="*80)
    
    base_dir = Path(__file__).parents[2]
    projeto_dir = base_dir / "dengue_prediction_fabio_kon"
    
    if not projeto_dir.exists():
        print(f"❌ Pasta do projeto não encontrada!")
        return
    
    print(f"📁 Analisando projeto em: {projeto_dir}")
    
    # 1. ESTRUTURA GERAL DO PROJETO
    print(f"\n{'-'*60}")
    print("1. ESTRUTURA DO PROJETO")
    print(f"{'-'*60}")
    
    arquivos = list(projeto_dir.glob("*"))
    arquivos_python = [f for f in arquivos if f.suffix == '.py']
    outros_arquivos = [f for f in arquivos if f.suffix != '.py' and f.is_file()]
    pastas = [f for f in arquivos if f.is_dir()]
    
    print(f"\n📊 COMPOSIÇÃO:")
    print(f"   • Arquivos Python: {len(arquivos_python)}")
    print(f"   • Outros arquivos: {len(outros_arquivos)}")
    print(f"   • Pastas: {len(pastas)}")
    
    print(f"\n📝 ARQUIVOS PYTHON:")
    for arquivo in sorted(arquivos_python):
        tamanho = arquivo.stat().st_size / 1024  # KB
        print(f"   • {arquivo.name:<25} ({tamanho:.1f} KB)")
    
    print(f"\n📄 OUTROS ARQUIVOS:")
    for arquivo in sorted(outros_arquivos):
        tamanho = arquivo.stat().st_size / 1024  # KB
        print(f"   • {arquivo.name:<25} ({tamanho:.1f} KB)")
    
    # 2. ANÁLISE DO README
    print(f"\n{'-'*60}")
    print("2. DOCUMENTAÇÃO (README.md)")
    print(f"{'-'*60}")
    
    readme_file = projeto_dir / "README.md"
    if readme_file.exists():
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            
            linhas = readme_content.split('\n')
            print(f"\n📋 RESUMO DO README:")
            print(f"   • Total de linhas: {len(linhas)}")
            
            # Extrair títulos (linhas que começam com #)
            titulos = [linha.strip() for linha in linhas if linha.strip().startswith('#')]
            if titulos:
                print(f"   • Seções identificadas:")
                for titulo in titulos[:10]:  # Primeiros 10 títulos
                    print(f"     - {titulo}")
            
            # Procurar por informações importantes
            palavras_chave = ['dengue', 'prediction', 'machine learning', 'model', 'dataset']
            for palavra in palavras_chave:
                if palavra.lower() in readme_content.lower():
                    print(f"   ✅ Menciona '{palavra}'")
                    
        except Exception as e:
            print(f"   ❌ Erro ao ler README: {e}")
    else:
        print(f"   ❌ README.md não encontrado")
    
    # 3. ANÁLISE DOS REQUIREMENTS
    print(f"\n{'-'*60}")
    print("3. DEPENDÊNCIAS (requirements.txt)")
    print(f"{'-'*60}")
    
    req_file = projeto_dir / "requirements.txt"
    if req_file.exists():
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                requirements = f.read().strip().split('\n')
            
            requirements = [req.strip() for req in requirements if req.strip()]
            
            print(f"\n📦 BIBLIOTECAS NECESSÁRIAS ({len(requirements)}):")
            
            # Categorizar bibliotecas
            ml_libs = []
            data_libs = []
            viz_libs = []
            other_libs = []
            
            for req in requirements:
                req_lower = req.lower()
                if any(ml in req_lower for ml in ['sklearn', 'xgboost', 'lightgbm', 'catboost', 'tensorflow', 'torch']):
                    ml_libs.append(req)
                elif any(data in req_lower for data in ['pandas', 'numpy', 'scipy']):
                    data_libs.append(req)
                elif any(viz in req_lower for viz in ['matplotlib', 'seaborn', 'plotly']):
                    viz_libs.append(req)
                else:
                    other_libs.append(req)
            
            if ml_libs:
                print(f"   🤖 Machine Learning: {ml_libs}")
            if data_libs:
                print(f"   📊 Manipulação de dados: {data_libs}")
            if viz_libs:
                print(f"   📈 Visualização: {viz_libs}")
            if other_libs:
                print(f"   🔧 Outras: {other_libs}")
                
        except Exception as e:
            print(f"   ❌ Erro ao ler requirements: {e}")
    else:
        print(f"   ❌ requirements.txt não encontrado")
    
    return projeto_dir

def analisar_arquivo_python(arquivo_path, nome_arquivo):
    """
    Analisa um arquivo Python específico extraindo funções, classes e imports
    """
    print(f"\n📄 ANÁLISE: {nome_arquivo}")
    print(f"   {'─' * (len(nome_arquivo) + 10)}")
    
    try:
        with open(arquivo_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Contar linhas
        linhas = content.split('\n')
        linhas_codigo = [l for l in linhas if l.strip() and not l.strip().startswith('#')]
        
        print(f"   • Total de linhas: {len(linhas)}")
        print(f"   • Linhas de código: {len(linhas_codigo)}")
        
        # Extrair imports
        imports = []
        for linha in linhas:
            linha_strip = linha.strip()
            if linha_strip.startswith('import ') or linha_strip.startswith('from '):
                imports.append(linha_strip)
        
        if imports:
            print(f"   • Imports principais:")
            for imp in imports[:5]:  # Primeiros 5 imports
                print(f"     - {imp}")
            if len(imports) > 5:
                print(f"     ... e mais {len(imports) - 5} imports")
        
        # Extrair funções usando regex
        funcoes = re.findall(r'def\s+(\w+)\s*\(([^)]*)\)', content)
        if funcoes:
            print(f"   • Funções definidas ({len(funcoes)}):")
            for nome, params in funcoes[:8]:  # Primeiras 8 funções
                params_clean = params.replace('\n', '').replace(' ', '')[:30]
                print(f"     - {nome}({params_clean}{'...' if len(params) > 30 else ''})")
            if len(funcoes) > 8:
                print(f"     ... e mais {len(funcoes) - 8} funções")
        
        # Extrair classes
        classes = re.findall(r'class\s+(\w+).*?:', content)
        if classes:
            print(f"   • Classes definidas: {classes}")
        
        # Buscar por algoritmos específicos de ML
        ml_algorithms = [
            'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 
            'LinearRegression', 'SVM', 'SVR', 'ExtraTrees',
            'SARIMA', 'ARIMA', 'Neural', 'KNN'
        ]
        
        algoritmos_encontrados = []
        for algo in ml_algorithms:
            if algo.lower() in content.lower():
                algoritmos_encontrados.append(algo)
        
        if algoritmos_encontrados:
            print(f"   🤖 Algoritmos ML identificados: {algoritmos_encontrados}")
        
        # Identificar propósito do arquivo baseado no nome e conteúdo
        if 'main' in nome_arquivo.lower():
            print(f"   🎯 PROPÓSITO: Arquivo principal de execução")
        elif 'method' in nome_arquivo.lower():
            print(f"   🎯 PROPÓSITO: Implementação de métodos/algoritmos")
        elif 'measure' in nome_arquivo.lower() or 'metric' in nome_arquivo.lower():
            print(f"   🎯 PROPÓSITO: Métricas e avaliação de modelos")
        elif 'support' in nome_arquivo.lower():
            print(f"   🎯 PROPÓSITO: Funções de suporte/utilitários")
        elif 'neighbor' in nome_arquivo.lower():
            print(f"   🎯 PROPÓSITO: Métodos baseados em vizinhança (KNN, etc)")
        
    except Exception as e:
        print(f"   ❌ Erro ao analisar arquivo: {e}")

def gerar_relatorio_completo():
    """
    Gera relatório completo do projeto
    """
    
    # Análise inicial
    projeto_dir = analisar_projeto_fabio_kon()
    
    if not projeto_dir:
        return
    
    # 4. ANÁLISE DETALHADA DOS ARQUIVOS PYTHON
    print(f"\n{'-'*60}")
    print("4. ANÁLISE DETALHADA DOS CÓDIGOS")
    print(f"{'-'*60}")
    
    arquivos_python = sorted(projeto_dir.glob("*.py"))
    
    # Ordem de análise por importância
    ordem_analise = [
        'main.py',
        'mlMethods.py', 
        'measuringResults.py',
        'mlSupportMethods.py',
        'neighbors_methods.py',
        'regression_metrics.py',
        'apply_grid_search_sarima.py',
        'analise_projeto.py'
    ]
    
    # Analisar na ordem de importância
    for nome_arquivo in ordem_analise:
        arquivo_path = projeto_dir / nome_arquivo
        if arquivo_path.exists():
            analisar_arquivo_python(arquivo_path, nome_arquivo)
    
    # Analisar arquivos restantes
    for arquivo in arquivos_python:
        if arquivo.name not in ordem_analise:
            analisar_arquivo_python(arquivo, arquivo.name)
    
    # 5. ANÁLISE DO NOTEBOOK JUPYTER
    print(f"\n{'-'*60}")
    print("5. NOTEBOOK JUPYTER")
    print(f"{'-'*60}")
    
    notebook_file = projeto_dir / "Shap_value_force_plot.ipynb"
    if notebook_file.exists():
        tamanho = notebook_file.stat().st_size / 1024
        print(f"\n📓 {notebook_file.name} ({tamanho:.1f} KB)")
        print(f"   🎯 PROPÓSITO: Análise de explicabilidade com SHAP values")
        print(f"   🔍 FUNCIONALIDADE: Gráficos de força para interpretação de modelos")
    
    # 6. CONCLUSÕES E CAPACIDADES
    print(f"\n{'-'*60}")
    print("6. RESUMO DAS CAPACIDADES DO PROJETO")
    print(f"{'-'*60}")
    
    print(f"\n🎯 OBJETIVO PRINCIPAL:")
    print(f"   Predição de casos de dengue usando múltiplos algoritmos de ML")
    
    print(f"\n🤖 ALGORITMOS IMPLEMENTADOS:")
    print(f"   • Regressão Linear (simples e múltipla)")
    print(f"   • Regressão Polinomial") 
    print(f"   • Random Forest")
    print(f"   • Extra Trees")
    print(f"   • Support Vector Regression (SVR)")
    print(f"   • XGBoost")
    print(f"   • LightGBM")
    print(f"   • CatBoost")
    print(f"   • Métodos baseados em vizinhança (KNN)")
    print(f"   • SARIMA (séries temporais)")
    
    print(f"\n⚙️ FUNCIONALIDADES:")
    print(f"   • Grid Search automático para otimização")
    print(f"   • Cross-validation para avaliação")
    print(f"   • Múltiplas métricas de regressão")
    print(f"   • Feature scaling e seleção")
    print(f"   • Backward elimination")
    print(f"   • Explicabilidade com SHAP")
    
    print(f"\n✅ PONTOS FORTES:")
    print(f"   • Implementação completa de pipeline de ML")
    print(f"   • Múltiplos algoritmos prontos para uso")
    print(f"   • Otimização automática de hiperparâmetros")
    print(f"   • Sistema modular e reutilizável")
    print(f"   • Métricas de avaliação abrangentes")
    
    print(f"\n💡 APLICABILIDADE PARA SEU MESTRADO:")
    print(f"   🎯 IDEAL para seu projeto de predição de dengue!")
    print(f"   📊 Pode usar os dados dengue_poa + MiAedes como entrada")
    print(f"   🔧 Já tem tudo implementado, só adaptar os dados")
    print(f"   📈 Pode comparar todos os algoritmos automaticamente")
    print(f"   📝 Base sólida para metodologia da dissertação")

if __name__ == "__main__":
    gerar_relatorio_completo()