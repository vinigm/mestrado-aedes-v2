#!/usr/bin/env python3
"""
Script para coletar e processar dados do MiAedes.
Coleta dados de mosquitos, processa e salva em Excel.
"""

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import logging
import argparse
from datetime import datetime
import os
import sys
from pathlib import Path
import time

# Importar configurações
try:
    from config import *
except ImportError:
    logging.error("Arquivo config.py não encontrado. Usando configurações padrão.")
    URL_BASE = 'https://www.miaedes.com.br/public-maps/client/72/region/72/weekly'
    HEADERS = {'User-Agent': 'Mozilla/5.0'}
    USAR_CACHE = False
    REQUEST_TIMEOUT = 30
    DEFAULT_OUTPUT = 'dados_aedes_{timestamp}.xlsx'
    CACHE_FILE = 'dados_cache.json'
    CACHE_DURATION = 3600

# Configurar logging
def setup_logging(verbose=False):
    """Configura o logging com formato e nível apropriados."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_cache():
    """Carrega dados do cache se existir e for válido."""
    if not USAR_CACHE or not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
            if time.time() - cache_data['timestamp'] < CACHE_DURATION:
                return cache_data['data']
    except Exception as e:
        logging.warning(f"Erro ao ler cache: {e}")
    return None

def save_cache(data):
    """Salva dados no cache com timestamp."""
    if not USAR_CACHE:
        return
    
    try:
        cache_data = {
            'timestamp': time.time(),
            'data': data
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        logging.warning(f"Erro ao salvar cache: {e}")

def fetch_data(url):
    """
    Busca dados da URL especificada.
    Retorna o conteúdo JSON processado.
    """
    # Tentar carregar do cache primeiro
    cached_data = load_cache()
    if cached_data:
        logging.info("Usando dados do cache")
        return cached_data

    logging.info(f"Acessando URL: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        html_content = response.text
        logging.debug("Página HTML baixada com sucesso")

        soup = BeautifulSoup(html_content, 'html.parser')
        script_tag = soup.find('script', {'id': '__NEXT_DATA__'})

        if not script_tag:
            raise ValueError("Tag <script> com id='__NEXT_DATA__' não encontrada")

        json_data = json.loads(script_tag.string)
        data = json_data['props']['pageProps']['data']
        
        # Validar estrutura dos dados
        if not isinstance(data, list):
            raise ValueError("Dados inválidos: esperava uma lista de registros")
        
        # Salvar no cache
        save_cache(data)
        
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro de conexão: {e}")
        raise
    except Exception as e:
        logging.error(f"Erro ao processar dados: {e}")
        raise

def process_data(lista_de_registros):
    """
    Processa a lista de registros e retorna um DataFrame limpo.
    """
    logging.info("Processando dados...")
    
    try:
        df_bruto = pd.DataFrame(lista_de_registros)
        logging.info(f"Processando {len(df_bruto)} registros")

        # Validar colunas necessárias
        required_cols = ['coordinates', 'address', 'inspection_mosquitos']
        missing_cols = [col for col in required_cols if col not in df_bruto.columns]
        if missing_cols:
            raise ValueError(f"Colunas ausentes nos dados: {missing_cols}")

        # Achatamento das colunas 'coordinates' e 'address'
        coordenadas_df = pd.json_normalize(df_bruto['coordinates'])
        endereco_df = pd.json_normalize(df_bruto['address'])

        # Processamento da coluna 'inspection_mosquitos'
        lista_inspecoes = df_bruto['inspection_mosquitos']

        # Dicionário de contagens por espécie e gênero
        especies = {
            'aedes_aegypti': 'Aedes aegypti',
            'aedes_albopictus': 'Aedes albopictus',
            'culex_sp': 'Culex sp'
        }

        # Criar colunas de contagem de forma mais eficiente
        for esp_key, esp_name in especies.items():
            for gender, suffix in [(1, 'femea'), (0, 'macho')]:
                col_name = f'{esp_key}_{suffix}'
                df_bruto[col_name] = lista_inspecoes.apply(
                    lambda x: sum(item['quantity'] for item in x 
                                if item['name'] == esp_name and item['gender'] == gender)
                )

        df_bruto['total_mosquitos'] = lista_inspecoes.apply(
            lambda x: sum(item['quantity'] for item in x)
        )

        # Limpeza e combinação final
        df_limpo = df_bruto.drop(['coordinates', 'address', 'inspection_mosquitos'], axis=1)
        df_final = pd.concat([df_limpo, coordenadas_df, endereco_df], axis=1)

        logging.info("Processamento concluído com sucesso")
        return df_final

    except Exception as e:
        logging.error(f"Erro no processamento dos dados: {e}")
        raise

def save_data(df, output_path):
    """
    Salva o DataFrame processado em Excel.
    """
    try:
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        df.to_excel(output_path, index=False)
        logging.info(f"Dados salvos em: {output_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar arquivo: {e}")
        raise

def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(description='Coleta e processa dados do MiAedes')
    parser.add_argument('-o', '--output', help='Caminho para salvar o arquivo Excel')
    parser.add_argument('-v', '--verbose', action='store_true', help='Modo verboso')
    parser.add_argument('--no-cache', action='store_true', help='Não usar cache')
    args = parser.parse_args()

    # Configurar logging
    setup_logging(args.verbose)

    try:
        # Desativar cache se solicitado
        global USAR_CACHE
        if args.no_cache:
            USAR_CACHE = False

        # Definir nome do arquivo de saída
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = args.output or DEFAULT_OUTPUT.format(timestamp=timestamp)

        # Executar pipeline de processamento
        logging.info("Iniciando coleta de dados")
        raw_data = fetch_data(URL_BASE)
        df_final = process_data(raw_data)
        save_data(df_final, output_path)

        # Mostrar resumo
        logging.info("\nResumo dos dados coletados:")
        logging.info(f"Total de registros: {len(df_final)}")
        logging.info(f"Total de mosquitos: {df_final['total_mosquitos'].sum()}")
        logging.info("Processo concluído com sucesso!")

    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
