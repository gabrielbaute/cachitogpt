from pathlib import Path
from typing import Dict

__version__ = "0.0.0"

class Config:
    """
    Clase de configuración para el proyecto.
    """
    # Información del modelo
    MODEL_NAME: str = "Cachito-GPT"
    MODEL_VERSION: str = __version__
    DESCRIPTION: str = "Sistema Generativo Local"

    # Obtener la ruta base del proyecto (la carpeta que contiene 'app')
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Rutas de los directorios
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    TOKENIZER_DIR: Path = BASE_DIR / "tokenizer"
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoint"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Asegurarse de que los directorios existan
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Rutas de archivos específicos
    TOKENIZER_PATH: Path = TOKENIZER_DIR / "tokenizer.json"
    MODEL_WEIGHTS_PATH: Path = MODEL_DIR / "gpt_model_pesos.pth"
    TRAINING_DATA_PATH: Path = DATA_DIR / "prueba.txt" # O el nombre que uses para tu corpus

    # GPT
    VOCAB_SIZE: int = 5000
    D_MODEL: int = 256     # 256 Min
    N_LAYERS: int = 2      # 2 Min
    NUM_HEADS: int = 8
    MAX_SEQ_LEN: int = 64  # Longitud de cada "frase" de entrenamiento (64 Min)
    BATCH_SIZE: int = 8    # Cuántas frases procesamos a la vez (8 Min)
    EPOCHS: int = 5        # Cuántas veces recorreremos el libro entero (5 Min)
    LEARNING_RATE: float = 3e-4

    # ======= API ========
    
    # Rutas de interfaz para la API
    UI_DIR = Path(__file__).parent.parent / "ui"
    STATIC_DIR = UI_DIR / "static"
    INDEX_HTML = UI_DIR / "index.html"

    # Parámetros de la API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_ALLOW_ORIGINS: list = ["*"]
    API_ALLOW_METHODS: list = ["*"]
    API_ALLOW_HEADERS: list = ["*"]
    API_ALLOW_CREDENTIALS: bool = True
    API_DEBUG: bool = True

    @classmethod
    def preset_50_percent(cls) -> Dict[str, int]:
        # Propuesta de escalado moderado
        return {
                "D_MODEL": 384,  # Un 50% más de resolución
                "N_LAYERS": 6,    # Triplicamos la profundidad para captar lógica filosófica
                "NUM_HEADS": 8,   # (384 es divisible por 8, d_k sería 48)
                "MAX_SEQ_LEN": 128 # Duplicamos la ventana de atención (fundamental para coherencia)
            }

    @classmethod
    def create_dirs(cls) -> bool:
        """
        Crea los directorios si no existen.
        """
        try:
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            cls.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
            cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error al crear directorios: {e}")
            return False