import logging
from pathlib import Path
from app.settings import Config, setup_logging

setup_logging()
logger = logging.getLogger("CLEAN")

def clean_checkpoint_dir(logger: logging.Logger) -> bool:
    """
    Elimina todos los archivos dentro del directorio de checkpoints.
    """
    checkpoint_dir: Path = Config.CHECKPOINT_DIR
    if not checkpoint_dir.exists():
        logger.warning(f"El directorio de checkpoints no existe: {checkpoint_dir}")
        return False
    
    files = list(checkpoint_dir.glob('*'))
    if not files:
        logger.info("No se encontraron archivos en el directorio de checkpoints.")
        return False
    
    try:
        for file in files:
            if file.is_file():
                file.unlink()
        logger.info(f"Eliminados {len(files)} archivos en el directorio de checkpoints.")
        return True
    
    except Exception as e:
        logger.error(f"Error al limpiar el directorio de checkpoints: {e}")
        return False

if __name__ == "__main__":
    success = clean_checkpoint_dir(logger)