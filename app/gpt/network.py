import torch
import logging
import torch.nn as nn
from typing import Optional

class FeedForward(nn.Module):
    """Red neuronal de paso hacia adelante posicionada.
    
    Aplica dos transformaciones lineales con una activación no lineal GELU
    entre ellas. Se aplica a cada token de forma independiente.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1) -> None:
        """Inicializa las capas del FFN.
        
        Args:
            d_model: Dimensión de entrada y salida (ej. 256).
            d_ff: Dimensión de la capa oculta. Por defecto 4 * d_model.
            dropout: Probabilidad de regularización para evitar overfitting.
        """
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        # Capa de expansión: proyecta a un espacio de mayor dimensionalidad
        self.w_1: nn.Linear = nn.Linear(d_model, d_ff)
        
        # Activación GELU: Proporciona una curvatura suave necesaria para 
        # el aprendizaje de funciones complejas.
        self.activation: nn.GELU = nn.GELU()
        
        # Capa de proyección: devuelve el vector al espacio d_model
        self.w_2: nn.Linear = nn.Linear(d_ff, d_model)
        
        # Dropout: Técnica de regularización (apaga neuronas aleatoriamente)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso forward del bloque FFN.
        
        Args:
            x: Tensor de forma (batch, seq_len, d_model).
            
        Returns:
            Tensor procesado de la misma forma (batch, seq_len, d_model).
        """
        # 1. Expandir y aplicar no-linealidad
        # En este punto, para cada token pasamos de 256 a 1024 dimensiones
        self.logger.debug(f"Activating FFN...")
        x = self.activation(self.w_1(x))
        
        # 2. Regularizar
        self.logger.debug(f"Applying dropout...")
        x = self.dropout(x)
        
        # 3. Proyectar de vuelta
        # Volvemos a las 256 dimensiones originales
        self.logger.debug(f"Projecting back...")
        x = self.w_2(x)
        
        return x