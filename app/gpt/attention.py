import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    """Implementación del mecanismo de Multi-Head Attention para un Transformer.
    
    Esta clase divide el espacio de embedding en múltiples 'cabezas' para permitir
    que el modelo atienda a diferentes representaciones en paralelo.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8) -> None:
        """Inicializa la capa de atención.
        
        Args:
            d_model: Dimensión total del modelo (ej. 256).
            num_heads: Número de cabezas de atención (debe ser divisor de d_model).
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_k: int = d_model // num_heads # Dimensión por cabeza
        
        # Proyecciones lineales para Queries, Keys y Values
        self.w_q: nn.Linear = nn.Linear(d_model, d_model)
        self.w_k: nn.Linear = nn.Linear(d_model, d_model)
        self.w_v: nn.Linear = nn.Linear(d_model, d_model)
        
        # Proyección final de salida
        self.w_o: nn.Linear = nn.Linear(d_model, d_model)

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Paso forward de la atención multicanal.
        
        Args:
            x: Tensor de entrada (batch, seq_len, d_model).
            mask: Máscara opcional para evitar atención en ciertos tokens (padding o futuro).
            
        Returns:
            Tensor procesado de la misma forma que la entrada.
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Proyecciones lineales y división en cabezas
        self.logger.debug(f"Input shape: {x.shape}")
        # Transformamos de (B, S, D) a (B, S, H, D_K) y trasponemos a (B, H, S, D_K)
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # scores = (Q @ K^T) / sqrt(d_k)
        self.logger.debug("Calculating attention scores...")
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax para obtener las probabilidades de atención
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 3. Multiplicación por Values y concatenación
        # output = weights @ V
        self.logger.debug("Applying attention weights...")
        out = torch.matmul(attn_weights, v) # (B, H, S, D_K)
        
        # Re-combinar cabezas: (B, H, S, D_K) -> (B, S, D)
        self.logger.debug("Re-combining heads...")
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 4. Proyección final
        self.logger.debug("Final projection...")
        return self.w_o(out)