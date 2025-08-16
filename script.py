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

try:
    # ================= PARTE 1: COLETA DOS DADOS BRUTOS =================

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html_content = response.text
    print("2. Página HTML baixada com sucesso!")

    soup = BeautifulSoup(html_content, 'html.parser')
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})

    if not script_tag:
        raise Exception("Não foi possível encontrar a tag <script> com id='__NEXT_DATA__'.")

    print("3. Bloco de dados JSON encontrado dentro do HTML!")

    json_data = json.loads(script_tag.string)
    lista_de_registros = json_data['props']['pageProps']['data']

    df_bruto = pd.DataFrame(lista_de_registros)
    print(f"4. Dados brutos extraídos com sucesso! Total de {len(df_bruto)} registros.")

    # ================= PARTE 2: PROCESSAMENTO E LIMPEZA DOS DADOS =================

    print("\n5. Iniciando aprimoramento dos dados...")

    # Achatamento das colunas 'coordinates' e 'address'
    # Como os dados vêm direto do JSON, eles já são objetos, não precisamos do 'ast' aqui.
    coordenadas_df = pd.json_normalize(df_bruto['coordinates'])
    endereco_df = pd.json_normalize(df_bruto['address'])

    # Achatamento da coluna 'inspection_mosquitos'
    # Aqui os dados também já são listas, então podemos aplicar a lógica diretamente
    lista_inspecoes = df_bruto['inspection_mosquitos']

    df_bruto['aedes_aegypti_femea'] = lista_inspecoes.apply(lambda x: sum(item['quantity'] for item in x if item['name'] == 'Aedes aegypti' and item['gender'] == 1))
    df_bruto['aedes_aegypti_macho'] = lista_inspecoes.apply(lambda x: sum(item['quantity'] for item in x if item['name'] == 'Aedes aegypti' and item['gender'] == 0))
    df_bruto['aedes_albopictus_femea'] = lista_inspecoes.apply(lambda x: sum(item['quantity'] for item in x if item['name'] == 'Aedes albopictus' and item['gender'] == 1))
    df_bruto['aedes_albopictus_macho'] = lista_inspecoes.apply(lambda x: sum(item['quantity'] for item in x if item['name'] == 'Aedes albopictus' and item['gender'] == 0))
    df_bruto['culex_sp_femea'] = lista_inspecoes.apply(lambda x: sum(item['quantity'] for item in x if item['name'] == 'Culex sp' and item['gender'] == 1))
    df_bruto['culex_sp_macho'] = lista_inspecoes.apply(lambda x: sum(item['quantity'] for item in x if item['name'] == 'Culex sp' and item['gender'] == 0))
    df_bruto['total_mosquitos'] = lista_inspecoes.apply(lambda x: sum(item['quantity'] for item in x))

    print("   ... colunas de contagem de mosquitos criadas.")

    # Combinação final e limpeza
    df_limpo = df_bruto.drop(['coordinates', 'address', 'inspection_mosquitos'], axis=1)
    df_final = pd.concat([df_limpo, coordenadas_df, endereco_df], axis=1)
    print("   ... colunas combinadas e limpas.")

    # ================= PARTE 3: EXIBIÇÃO E DOWNLOAD =================

    print("\n✅ Processo concluído! Amostra da tabela final e tratada:")
    print(df_final.head())

    output_filename = 'dados_aedes_COMPLETO_E_TRATADO.xlsx'
    df_final.to_excel(output_filename, index=False)
    print(f"\nArquivo '{output_filename}' pronto para download.")

    files.download(output_filename)

except requests.exceptions.RequestException as e:
    print(f"Erro de conexão: {e}. O site pode estar fora do ar ou o bloqueio pode ter mudado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")