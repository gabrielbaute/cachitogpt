import torch
import torch.nn.functional as F

from app.settings import Config
from app.gpt import GPTModel, BPETokenizer

def generate():
    # 1. Cargar herramientas
    tokenizer = BPETokenizer.load(str(Config.TOKENIZER_PATH))
    model = GPTModel(Config.VOCAB_SIZE, Config.D_MODEL, Config.N_LAYERS, Config.NUM_HEADS, Config.MAX_SEQ_LEN)
    
    # Cargar los pesos entrenados
    model.load_state_dict(torch.load("./model/gpt_model_pesos.pth", weights_only=True))
    model.eval() # Modo evaluación: desactiva Dropout

    # 2. Parámetros de generación
    prompt = "Quién era Galileo?"
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor(tokens).unsqueeze(0) # [1, seq_len]
    
    max_new_tokens = 100
    temperature = 0.8  # Controla la creatividad (1.0 = normal, <1.0 = más conservador)

    print(f"--- Generando desde: '{prompt}' ---")
    
    generated = tokens
    for _ in range(max_new_tokens):
        # Recortamos el contexto si excede MAX_SEQ_LEN
        idx_cond = input_tensor[:, -Config.MAX_SEQ_LEN:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            # Solo nos interesa el último token predicho
            logits = logits[:, -1, :] / temperature
            
            # Convertimos a probabilidades
            probs = F.softmax(logits, dim=-1)
            
            # Muestreamos (en lugar de elegir siempre el más alto, para dar variedad)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Añadimos a la secuencia
            input_tensor = torch.cat((input_tensor, next_token), dim=1)
            generated.append(next_token.item())
            
            # Si el modelo generó un token de fin (si lo tuvieras), podrías parar aquí
            
    print(f"Resultado: {tokenizer.decode(generated)}")

if __name__ == "__main__":
    generate()