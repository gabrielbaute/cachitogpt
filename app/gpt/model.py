import torch
import logging
import torch.nn as nn
from app.gpt.embedding import TransformerEmbedding
from app.gpt.transformer_block import TransformerBlock

class GPTModel(nn.Module):
    """Arquitectura completa tipo GPT (Decoder-only Transformer)."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 1. Entrada: IDs -> Vectores + Posición
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len)
        
        # 2. Cuerpo: Pila de bloques Transformer
        # Usamos nn.ModuleList para que PyTorch registre correctamente las capas
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(n_layers)
        ])
        
        # 3. Salida: Capa de normalización final y proyección a vocabulario
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Paso forward del modelo.
        
        Args:
            input_ids: Tensor de forma (batch_size, sequence_length) con IDs de tokens.
            
        Returns:
            Tensor de forma (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # Crear la Máscara Causal (Look-ahead mask)
        # Es una matriz triangular de ceros y unos
        self.logger.debug("Creating causal mask...")
        mask = torch.tril(torch.ones((seq_len, seq_len))).view(1, 1, seq_len, seq_len)
        
        # Flujo
        x = self.embedding(input_ids)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, S, Vocab_Size)
        
        return logits