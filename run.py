import logging
from pathlib import Path
from app.generator import TextGenerator
from app.settings import setup_logging, Banner, Config

setup_logging(model_name="dussel")
logger_generator = logging.getLogger("TextGenerator")

def main():
    generator = TextGenerator(model_name="dussel")
    banner = Banner(name=Config.MODEL_NAME, version=Config.MODEL_VERSION)
    print(f"{banner.get_banner()}")
    print("Escribe 'salir' para terminar.")
    
    while True:
        prompt = input("Tú (Prompt) -> ")
        if prompt.lower() == 'salir':
            break
            
        try:
            resultado = generator.generate_with_top_filters(
                prompt,
                max_new_tokens=100,
                temperature=1.2,
                top_k=100,
                top_p=0.95
            )
            print(f"\n{Config.MODEL_NAME} -> {resultado.texto_generado}\n")
        except Exception as e:
            logger_generator.error(f"Error en generación: {e}")

if __name__ == "__main__":
    main()