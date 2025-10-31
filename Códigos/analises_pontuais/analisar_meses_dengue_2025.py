#!/usr/bin/env python3
"""
Análise pontual: Verificar distribuição mensal dos casos de dengue em DENGBR25.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def analisar_meses_dengue():
    """
    Analisa a distribuição mensal dos casos de dengue em DENGBR25.csv
    """
    
    print("="*80)
    print(" ANÁLISE: MESES COM DADOS DE DENGUE EM 2025 ".center(80, "="))
    print("="*80)
    
    base_dir = Path(__file__).parents[2]
    pasta_oficiais = base_dir / "Bases de dados" / "bases_oficiais_opendatasus"
    arquivo_dengue = pasta_oficiais / "DENGBR25.csv"
    
    if not arquivo_dengue.exists():
        print(f"❌ Arquivo DENGBR25.csv não encontrado!")
        return
    
    print(f"📁 Analisando: {arquivo_dengue.name}")
    
    # 1. ANÁLISE POR CHUNKS PARA ARQUIVOS GRANDES
    print(f"\n🔍 Carregando e analisando dados...")
    
    try:
        chunk_size = 50000
        semanas_casos = {}
        meses_casos = {}
        total_registros = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(arquivo_dengue, chunksize=chunk_size, low_memory=False), 1):
            total_registros += len(chunk)
            
            # Analisar semanas epidemiológicas
            if 'SEM_NOT' in chunk.columns:
                semanas_chunk = chunk['SEM_NOT'].value_counts()
                for semana, freq in semanas_chunk.items():
                    semanas_casos[semana] = semanas_casos.get(semana, 0) + freq
            
            # Extrair mês da data de notificação se disponível
            if 'DT_NOTIFIC' in chunk.columns:
                # Converter datas e extrair mês
                try:
                    chunk['DT_NOTIFIC'] = pd.to_datetime(chunk['DT_NOTIFIC'], errors='coerce')
                    chunk['MES'] = chunk['DT_NOTIFIC'].dt.month
                    meses_chunk = chunk['MES'].value_counts()
                    for mes, freq in meses_chunk.items():
                        if pd.notna(mes):  # Ignorar valores nulos
                            meses_casos[int(mes)] = meses_casos.get(int(mes), 0) + freq
                except:
                    pass
            
            # Mostrar progresso a cada 10 chunks
            if chunk_num % 10 == 0:
                print(f"   Processados {total_registros:,} registros...")
        
        print(f"✅ Total analisado: {total_registros:,} registros")
        
        # 2. ANÁLISE POR SEMANAS EPIDEMIOLÓGICAS
        print(f"\n{'-'*60}")
        print("📅 DISTRIBUIÇÃO POR SEMANA EPIDEMIOLÓGICA 2025")
        print(f"{'-'*60}")
        
        if semanas_casos:
            semanas_ordenadas = sorted(semanas_casos.items())
            
            print(f"\n📊 Semanas com dados (primeiras e últimas):")
            
            # Mostrar primeiras 10 semanas
            print(f"\n🟢 PRIMEIRAS SEMANAS:")
            for semana, casos in semanas_ordenadas[:10]:
                # Converter semana epidemiológica para mês aproximado
                semana_str = str(semana)
                if len(semana_str) == 6:  # Formato AAAASS
                    num_semana = int(semana_str[-2:])
                    mes_aprox = (num_semana - 1) // 4 + 1  # Aproximação: 4 semanas = 1 mês
                    mes_aprox = min(mes_aprox, 12)
                    print(f"   • Semana {semana} (~mês {mes_aprox:2d}): {casos:,} casos")
                else:
                    print(f"   • Semana {semana}: {casos:,} casos")
            
            # Mostrar últimas 10 semanas
            print(f"\n🔴 ÚLTIMAS SEMANAS:")
            for semana, casos in semanas_ordenadas[-10:]:
                semana_str = str(semana)
                if len(semana_str) == 6:
                    num_semana = int(semana_str[-2:])
                    mes_aprox = (num_semana - 1) // 4 + 1
                    mes_aprox = min(mes_aprox, 12)
                    print(f"   • Semana {semana} (~mês {mes_aprox:2d}): {casos:,} casos")
                else:
                    print(f"   • Semana {semana}: {casos:,} casos")
            
            # Calcular cobertura mensal aproximada
            semanas_2025 = [s for s in semanas_casos.keys() if str(s).startswith('2025')]
            if semanas_2025:
                primeira_semana = min(semanas_2025)
                ultima_semana = max(semanas_2025)
                
                print(f"\n📈 RESUMO TEMPORAL:")
                print(f"   • Primeira semana: {primeira_semana}")
                print(f"   • Última semana: {ultima_semana}")
                print(f"   • Total de semanas: {len(semanas_2025)}")
                
                # Estimar meses com base nas semanas
                primeira_sem_num = int(str(primeira_semana)[-2:])
                ultima_sem_num = int(str(ultima_semana)[-2:])
                
                mes_inicio = (primeira_sem_num - 1) // 4 + 1
                mes_fim = (ultima_sem_num - 1) // 4 + 1
                
                print(f"   • Período estimado: mês {mes_inicio} até mês {mes_fim}")
        
        # 3. ANÁLISE POR DATA DE NOTIFICAÇÃO
        print(f"\n{'-'*60}")
        print("📅 DISTRIBUIÇÃO POR MÊS (DATA DE NOTIFICAÇÃO)")
        print(f"{'-'*60}")
        
        if meses_casos:
            meses_nomes = {
                1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
                5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
                9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
            }
            
            print(f"\n📊 Casos por mês em 2025:")
            total_casos = sum(meses_casos.values())
            
            for mes in sorted(meses_casos.keys()):
                casos = meses_casos[mes]
                percentual = (casos / total_casos) * 100
                print(f"   • {meses_nomes[mes]:10s} (mês {mes:2d}): {casos:6,} casos ({percentual:5.1f}%)")
            
            print(f"\n📈 RESUMO:")
            print(f"   • Total de casos: {total_casos:,}")
            print(f"   • Meses com dados: {sorted(meses_casos.keys())}")
            print(f"   • Período: {meses_nomes[min(meses_casos.keys())]} até {meses_nomes[max(meses_casos.keys())]}")
            
        else:
            print(f"❌ Não foi possível extrair informações mensais das datas")
        
        # 4. CONCLUSÃO SOBRE COBERTURA TEMPORAL
        print(f"\n{'-'*60}")
        print("🎯 CONCLUSÃO SOBRE COBERTURA TEMPORAL")
        print(f"{'-'*60}")
        
        if meses_casos and semanas_casos:
            meses_com_dados = sorted(meses_casos.keys())
            semanas_com_dados = len([s for s in semanas_casos.keys() if str(s).startswith('2025')])
            
            print(f"\n✅ DADOS DISPONÍVEIS PARA:")
            print(f"   • Meses: {meses_com_dados}")
            print(f"   • Semanas epidemiológicas: {semanas_com_dados} semanas de 2025")
            
            # Verificar se dados chegam até outubro (mês atual)
            if 10 in meses_com_dados:
                print(f"   🟢 Dados atualizados até outubro (mês atual)")
            elif max(meses_com_dados) >= 8:
                print(f"   🟡 Dados atualizados até {meses_nomes[max(meses_com_dados)]} (alguns meses de atraso)")
            else:
                print(f"   🟠 Dados com atraso significativo (último mês: {meses_nomes[max(meses_com_dados)]})")
            
            print(f"\n💡 INTERPRETAÇÃO:")
            if len(meses_com_dados) >= 8:
                print(f"   • Boa cobertura temporal para análises sazonais")
                print(f"   • Suficiente para correlação com dados MiAedes")
            else:
                print(f"   • Cobertura limitada ({len(meses_com_dados)} meses)")
                print(f"   • Pode impactar análises de tendências anuais")
        
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")

if __name__ == "__main__":
    analisar_meses_dengue()