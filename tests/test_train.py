import logging
from pathlib import Path
from app.trainer import TrainModule
from app.settings import Config, setup_logging

def main():
    # 1. Configuración de logs para ver qué ocurre internamente
    setup_logging(model_name="training")
    logger = logging.getLogger("TestValidation")

    # 2. Definir ruta del dataset pequeño
    dataset_small: Path = Config.DATA_DIR / "corpus_2.txt"
    
    if not dataset_small.exists():
        logger.error(f"No se encontró el archivo en {dataset_small}")
        return

    # 3. Instanciar el Trainer
    # Usamos la configuración de Config directamente
    trainer = TrainModule(config=Config, dataset_path=dataset_small, model_name="ortegagasset2", stride=8)

    # 4. (Opcional) Intentar cargar un checkpoint si quieres probar esa lógica
    # trainer.load_checkpoint(Config.MODEL_DIR / "checkpoint_epoch_2.pth")

    try:
        logger.info("Iniciando prueba de validación corta...")
        # Ejecutamos el entrenamiento
        trainer.train()
        logger.info("¡Validación exitosa! El ciclo de entrenamiento completó correctamente.")
        
    except Exception as e:
        logger.error(f"Error durante la validación: {e}", exc_info=True)

if __name__ == "__main__":
    main()