import torch
import logging
import torch.nn as nn
from typing import Optional
from app.gpt.attention import MultiHeadAttention
from app.gpt.network import FeedForward

class TransformerBlock(nn.Module):
    """Un bloque completo de Transformer (Decodificador).
    
    Combina Atención Multicabezal, Feed-Forward y Normalización de Capa
    con conexiones residuales.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None, dropout: float = 0.1) -> None:
        """Inicializa los componentes del bloque.
        
        Args:
            d_model: Dimensión del modelo (ej. 256).
            num_heads: Número de cabezas de atención.
            d_ff: Dimensión oculta del FFN (4 * d_model).
            dropout: Ratio de regularización.
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Componente 1: Atención y su normalización
        self.attention: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        
        # Componente 2: Feed-Forward y su normalización
        self.feed_forward: FeedForward = FeedForward(d_model, d_ff, dropout)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Paso forward del bloque.
        
        Aplicamos el patrón: x = x + Dropout(Sublayer(LayerNorm(x)))
        Este es el orden 'Pre-Norm', preferido en modelos modernos por su estabilidad.
        """
        
        # Subcapa 1: Atención
        # 1. Normalizamos antes de entrar (Pre-Norm)
        norm_x = self.norm1(x)
        # 2. Calculamos atención
        attn_out = self.attention(norm_x, mask)
        # 3. Suma residual
        x = x + self.dropout(attn_out)
        
        # Subcapa 2: Feed-Forward
        # 1. Normalizamos
        norm_x = self.norm2(x)
        # 2. Procesamos con FFN
        ff_out = self.feed_forward(norm_x)
        # 3. Suma residual
        x = x + self.dropout(ff_out)
        
        self.logger.debug(f"Transformer block output shape: {x.shape}")
        return x