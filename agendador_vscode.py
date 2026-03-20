#!/usr/bin/env python3
"""
Agendador automático para executar raspagem.py diariamente às 20h.
Este script deve ficar rodando no VS Code em background.
"""

import schedule
import time
import subprocess
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Configurar diretório e logging
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configurar logging
log_file = LOG_DIR / f"agendador_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def executar_scraping():
    """Executa o script de raspagem."""
    try:
        logging.info("🕐 Executando scraping agendado...")
        
        script_path = SCRIPT_DIR / "raspagem.py"
        
        # Determinar comando Python (Linux: python3, Windows: py)
        if sys.platform.startswith('linux') or sys.platform == 'darwin':
            python_cmd = "python3"
        else:
            python_cmd = "py"
        
        # Verificar se o comando existe
        try:
            subprocess.run([python_cmd, "--version"], 
                         capture_output=True, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: tentar o outro comando
            python_cmd = "py" if python_cmd == "python3" else "python3"
            logging.info(f"⚠️ Usando comando alternativo: {python_cmd}")
        
        result = subprocess.run(
            [python_cmd, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR)
        )
        
        if result.returncode == 0:
            logging.info("✅ Scraping concluído com sucesso!")
            if result.stdout:
                logging.info(f"Saída: {result.stdout}")
        else:
            logging.error(f"❌ Erro no scraping: {result.stderr}")
            
    except Exception as e:
        logging.error(f"❌ Erro ao executar scraping: {str(e)}")

def main():
    """Inicia o agendador."""
    logging.info("="*60)
    logging.info("🚀 Agendador iniciado!")
    logging.info(f"📂 Diretório: {SCRIPT_DIR}")
    logging.info(f"🐍 Sistema: {sys.platform}")
    logging.info("⏰ Scraping agendado para: 20:00 todos os dias")
    logging.info("="*60)
    
    # Agendar para 20:00 todos os dias
    schedule.every().day.at("20:00").do(executar_scraping)
    
    # Loop infinito verificando os agendamentos
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verifica a cada 1 minuto
            
    except KeyboardInterrupt:
        logging.info("\n⏹️ Agendador finalizado pelo usuário")

if __name__ == "__main__":
    main()
