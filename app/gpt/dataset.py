import torch
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional
from torch.utils.data import Dataset
from app.gpt.tokenizer import BPETokenizer

class TextDataset(Dataset):
    """
    Dataset optimizado para entrenamiento de lenguaje.
    Soporta múltiples archivos y reduce la redundancia mediante saltos de bloque.
    """

    def __init__(self, 
                 path: Union[str, Path], 
                 tokenizer: BPETokenizer, 
                 seq_len: int,
                 stride: Optional[int] = None) -> None:
        """
        Args:
            path: Ruta a un archivo .txt o a una carpeta que contenga varios .txt.
            tokenizer: Instancia del tokenizer.
            seq_len: Longitud de la secuencia de contexto.
            stride: Desplazamiento entre ejemplos. Si es None, se usa seq_len 
                    (bloques no solapados), lo que acelera el entrenamiento x64 veces.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 1. Cargar y concatenar texto de uno o varios archivos
        all_text = self._load_corpus(path)
        
        # 2. Tokenizar el corpus completo
        self.tokens: List[int] = self.tokenizer.encode(all_text)
        self.logger.info(f"Corpus procesado: {len(self.tokens)} tokens encontrados.")

    def _load_corpus(self, path: Union[str, Path]) -> str:
        """Lee uno o varios archivos y los une en un solo string."""
        path = Path(path)
        texts = []
        
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob("*.txt"))
            
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        
        # Unimos con un separador de documento si el tokenizer lo soporta
        return "\n\n".join(texts)

    def __len__(self) -> int:
        """Calcula el número de bloques disponibles según el stride."""
        if len(self.tokens) <= self.seq_len:
            return 0
        return (len(self.tokens) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtiene un par (X, Y) usando el desplazamiento (stride)."""
        start_idx = idx * self.stride
        chunk = self.tokens[start_idx : start_idx + self.seq_len + 1]
        
        # Convertir a tensores
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y