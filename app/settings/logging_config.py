import logging
from app.settings.config import Config

def setup_logging() -> None:
    """
    Función para establecer la configuración de registro básica.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(str(Config.LOG_DIR / "app.log"), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
