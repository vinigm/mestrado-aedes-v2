#!/usr/bin/env python3
"""Executor automático do scraping MiAedes."""

import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

def main():
    """Executa o scraping automaticamente."""
    # CRUCIAL: Garantir que estamos no diretório correto
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Configurar logging
    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"scraping_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    try:
        script_path = script_dir / "raspagem.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script não encontrado: {script_path}")
        
        logging.info(f"Diretório de trabalho: {os.getcwd()}")
        logging.info("Iniciando scraping...")
        
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, cwd=str(script_dir))
        
        if result.returncode == 0:
            logging.info("✅ Scraping concluído com sucesso")
        else:
            logging.error(f"❌ Erro no scraping: {result.stderr}")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"❌ Erro: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()