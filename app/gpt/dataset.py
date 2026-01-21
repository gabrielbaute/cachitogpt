import torch
import logging
from typing import List, Tuple
from torch.utils.data import Dataset
from app.gpt.tokenizer import BPETokenizer

class TextDataset(Dataset):
    """Dataset para entrenamiento de lenguaje autorregresivo.
    
    Carga un archivo de texto, lo tokeniza y prepara pares de 
    entrada/objetivo (X, Y) desplazados por un token.
    """

    def __init__(self, file_path: str, tokenizer: BPETokenizer, seq_len: int) -> None:
        """Inicializa el dataset.
        
        Args:
            file_path: Ruta al archivo .txt.
            tokenizer: Instancia del tokenizer ya entrenado.
            seq_len: Longitud de la secuencia para el modelo (context window).
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Leemos el archivo y lo convertimos a una sola lista gigante de IDs
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenizamos todo el corpus de una vez
        self.tokens: List[int] = self.tokenizer.encode(text)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Dataset cargado: {len(self.tokens)} tokens encontrados.")

    def __len__(self) -> int:
        """Devuelve la cantidad de fragmentos disponibles."""
        # Restamos seq_len porque cada ejemplo necesita seq_len tokens + 1 para el target
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtiene un par de entrenamiento (X, Y)."""
        # Extraemos la secuencia de tamaño seq_len
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        
        # X: del token 0 al penúltimo
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # Y: del token 1 al último (la 'respuesta' desplazada)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y