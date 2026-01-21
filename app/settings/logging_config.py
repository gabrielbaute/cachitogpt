import logging
from typing import Optional
from app.settings.config import Config

def setup_logging(model_name: Optional[str] = None) -> None:
    """
    Función para establecer la configuración de registro básica.
    """
    log_file_name: str = f"{Config.MODEL_NAME if not model_name else model_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(str(Config.LOG_DIR / log_file_name), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
