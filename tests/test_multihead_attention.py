import torch
from app.gpt import BPETokenizer, TransformerEmbedding, MultiHeadAttention

def run_pipeline_test():
    """Ejecuta el flujo completo incorporando Multi-Head Attention."""
    
    # 1. Hiperparámetros
    VOCAB_SIZE = 5000
    D_MODEL = 256
    MAX_SEQ_LEN = 128
    NUM_HEADS = 8  # Cada cabeza tendrá dimensión 32 (256/8)
    
    # 2. Tokenizer (ya entrenado en el paso anterior)
    print("--- 1. Preparando Datos ---")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    # Reusamos el corpus.txt que ya creamos
    tokenizer.train(["corpus_2.txt"])
    # Guardamos el tokenizer
    tokenizer.save("./tokenizer.json")

    input_text = "Aprender Python es genial"
    tokens_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor(tokens_ids).unsqueeze(0) 
    print(f"Frase: {input_text}")
    print(f"Tokens: {tokenizer.decode(tokens_ids)}")
    print(f"Tensor: {input_tensor}")

    # 3. Inicialización de Capas
    embedding_layer = TransformerEmbedding(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN)
    attention_layer = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    
    # 4. Flujo de Datos (Forward Pass)
    print("\n--- 2. Ejecutando Tensores ---")
    with torch.no_grad():
        # Paso A: Texto -> Vectores (Embedding)
        embedded_vectors = embedding_layer(input_tensor)
        print(f"Shape tras Embedding: {embedded_vectors.shape}")
        
        # Paso B: Vectores -> Atención (MHA)
        # Por ahora no usamos máscara para ver el proceso completo
        attended_vectors = attention_layer(embedded_vectors)
        print(f"Shape tras Atención: {attended_vectors.shape}")

    # 5. Verificación de la Transformación
    print("\n--- 3. Verificación de Pesos ---")
    # Comparamos el primer elemento del primer token antes y después
    print(f"Valor original (Emb): {embedded_vectors[0, 0, 0].item():.4f}")
    print(f"Valor procesado (Attn): {attended_vectors[0, 0, 0].item():.4f}")

if __name__ == "__main__":
    run_pipeline_test()