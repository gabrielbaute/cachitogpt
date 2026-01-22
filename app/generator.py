import logging
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from app.gpt import GPTModel, BPETokenizer
from app.schemas.model_response import CachitoResponse
from app.settings import Config

class TextGenerator:
    """Clase para gestionar la inferencia y generación de texto cargando configuración desde JSON."""
    
    def __init__(self, model_name: str = "cachito"):
        """
        Inicializa el generador cargando automáticamente la arquitectura desde el JSON.
        
        Args:
            model_name (str): Nombre del modelo (sin extensión) a cargar.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        
        # 1. Definir rutas basadas en el nombre del modelo
        self.weights_path = Config.MODEL_DIR / f"{model_name}.pth"
        self.config_path = Config.MODEL_DIR / f"{model_name}_config.json"
        
        # 2. Cargar Metadatos desde JSON
        self.model_config = self._load_model_config()
        
        # 3. Cargar Tokenizer (el path sigue viniendo de Config por ahora)
        self.tokenizer = BPETokenizer.load(str(Config.TOKENIZER_PATH))
        
        # 4. Instanciar el modelo con la arquitectura exacta del entrenamiento
        arch = self.model_config["architecture"]
        self.model = GPTModel(
            vocab_size=arch["vocab_size"], 
            d_model=arch["dimension_model"] if "dimension_model" in arch else arch["d_model"], 
            n_layers=arch["number_layers"] if "number_layers" in arch else arch["n_layers"], 
            num_heads=arch["number_heads"] if "number_heads" in arch else arch["num_heads"], 
            max_seq_len=arch["max_seq_len"]
        )
        self.max_seq_len = arch["max_seq_len"]
        
        # 5. Cargar pesos
        if self.weights_path.exists():
            self.model.load_state_dict(torch.load(self.weights_path, weights_only=True))
            self.model.eval()
            self.logger.info(f"Modelo '{model_name}' cargado exitosamente ({arch['total_params']} parámetros).")
        else:
            raise FileNotFoundError(f"No se encontraron los pesos en {self.weights_path}")

    def _load_model_config(self) -> dict:
        """Lee el archivo JSON de configuración del modelo."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de configuración: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_with_top_filters(
            self,
            prompt: str,
            max_new_tokens: int = 100,
            temperature: float = 0.8,
            top_k: int = 40,
            top_p: float = 0.9,
            repetition_penalty: float = 1.2
        ) -> CachitoResponse:
        """
        Genera texto con filtros avanzados. 
        Nota: Usa self.max_seq_len cargado del JSON para el contexto.
        """
        tokens = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(tokens).unsqueeze(0)
        generated = tokens

        for _ in range(max_new_tokens):
            # Usamos el contexto exacto definido en el JSON
            idx_cond = input_tensor[:, -self.max_seq_len:]
            
            with torch.no_grad():
                logits = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Penalización por Repetición
                for token_id in set(generated):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

                # Top-K
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-P
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_tensor = torch.cat((input_tensor, next_token), dim=1)
                generated.append(next_token.item())
                
                if next_token.item() == self.tokenizer.tokenizer.token_to_id("[SEP]"):
                    break

        response = CachitoResponse(
            id=self.model_name,
            tokens_del_mensaje=len(tokens),
            tokens_generados=len(generated),
            texto_generado=self.tokenizer.decode(generated)
        ) 
        return response
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8):
        """
        Genera una secuencia de texto a partir de un prompt.
        
        Args:
            prompt(Str): Prompt o texto de entrada del usuario.
            max_new_tokens(int): Número máximo de tokens a generar.
            temperature(float): Parámetro de temperatura para controlar la aleatoriedad.
        
        Returns:
            Str: Texto generado por el modelo.
        """
        tokens = self.tokenizer.encode(prompt)
        self.logger.info(f"Codificando prompt, {len(tokens)} tokens obtenidos. Ventana de respuesta: {max_new_tokens} tokens.")
        input_tensor = torch.tensor(tokens).unsqueeze(0)
        self.logger.info(f"El modelo mantiene un contexto de {Config.MAX_SEQ_LEN} tokens.")
        generated = tokens
        for _ in range(max_new_tokens):
            # Mantener el contexto dentro de los límites del modelo
            idx_cond = input_tensor[:, -Config.MAX_SEQ_LEN:]
            
            with torch.no_grad():
                logits = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_tensor = torch.cat((input_tensor, next_token), dim=1)
                generated.append(next_token.item())
                
                # Opcional: Detener si genera un salto de línea o token de fin
                if next_token.item() == self.tokenizer.tokenizer.token_to_id("[SEP]"):
                    self.logger.info("Generación detenida por token [SEP].")
                    break

        self.logger.info(f"Devolviendo respuesta de {len(generated)} tokens.") 
        return self.tokenizer.decode(generated)