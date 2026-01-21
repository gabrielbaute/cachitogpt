import torch
from app.transformer_block import TransformerBlock

def test_internal_logic():
    """Verifica la integridad dimensional y numérica del bloque Transformer."""
    
    # Hiperparámetros de prueba
    batch_size = 1
    seq_len = 10
    d_model = 256
    num_heads = 8
    
    # 1. Instanciamos el bloque
    block = TransformerBlock(d_model=d_model, num_heads=num_heads)
    
    # 2. Creamos una entrada sintética (simulando la salida del Embedding)
    # Media 0, Desviación Estándar 1
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"--- Prueba de TransformerBlock ---")
    print(f"Input Shape: {x.shape}")
    
    # 3. Paso Forward
    block.eval() # Modo evaluación (desactiva Dropout)
    with torch.no_grad():
        output = block(x)
        
    # 4. Verificaciones
    print(f"Output Shape: {output.shape}")
    
    # Verificamos si hubo cambios drásticos en la magnitud de los datos
    print(f"Media Input: {x.mean().item():.4f} | Media Output: {output.mean().item():.4f}")
    print(f"Varianza Input: {x.std().item()**2:.4f} | Varianza Output: {output.std().item()**2:.4f}")
    
    assert x.shape == output.shape, "Error: La forma del tensor cambió tras el bloque."
    print("\n✅ El bloque preserva la dimensionalidad y estabilidad numérica.")

if __name__ == "__main__":
    test_internal_logic()