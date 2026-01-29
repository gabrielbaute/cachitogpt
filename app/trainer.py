import json
import torch
import logging
import torch.optim as optim
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

from app.gpt import GPTModel, BPETokenizer, TextDataset
from app.settings import Config

class TrainModule:
    """
    Encapsula la lógica de entrenamiento del modelo GPT.
    """
    def __init__(
            self, 
            config: Config, 
            dataset_path: Path, 
            model_name: Optional[str] = None,
            model_version: Optional[str] = None, 
            stride: Optional[int] = None
        ) -> None:
        """
        Inicializa el módulo de entrenamiento.

        Args:
            config(Config): Configuración del modelo.
            dataset_path(Path): Ruta al archivo de datos.
            model_name(Optional[str]): Nombre opcional del modelo.
            stride(Optional[int]): Desplazamiento entre ejemplos.
        """
        self.config = config
        self.dataset_path = dataset_path
        self.model_name = model_name if model_name else config.MODEL_NAME
        self.model_version = model_version if model_version else config.MODEL_VERSION
        self.stride = stride if stride is not None else config.MAX_SEQ_LEN
        self.checkpoint_dir = self.config.CHECKPOINT_DIR
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 1. Cargar Tokenizer
        self.tokenizer = BPETokenizer.load(str(self.config.TOKENIZER_PATH))
        
        # 2. Inicializar Modelo
        self.model = self._initialize_model()
        
        # 3. Componentes de Entrenamiento
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.start_epoch = 0

    def _initialize_model(self) -> GPTModel:
        self.logger.info("Instanciando el Modelo...")
        return GPTModel(
            vocab_size=self.config.VOCAB_SIZE,
            d_model=self.config.D_MODEL,
            n_layers=self.config.N_LAYERS,
            num_heads=self.config.NUM_HEADS,
            max_seq_len=self.config.MAX_SEQ_LEN
        )
    
    def _save_model_metadata(
              self, 
              final_loss: float, 
              dataset_size_chars: int, 
              model_path: Path,
              start_time: datetime,
              end_time: datetime
            ) -> None:
            """
            Genera y guarda un archivo JSON con los metadatos del modelo.

            Args:
                final_loss(float): Pérdida final del modelo.
                dataset_size_chars(int): Tamaño del dataset en caracteres.
                model_path(Path): Ruta donde se guardó el modelo.

            Returns:
                None
            """
            model_data = {
                "metadata": {
                    "model_name": self.model_name,
                    "version": self.model_version,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "device": str(next(self.model.parameters()).device),
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "training_duration": str(end_time - start_time)
                },
                "architecture": {
                    "vocab_size": self.config.VOCAB_SIZE,
                    "d_model": self.config.D_MODEL,
                    "n_layers": self.config.N_LAYERS,
                    "num_heads": self.config.NUM_HEADS,
                    "max_seq_len": self.config.MAX_SEQ_LEN,
                    "total_params": sum(p.numel() for p in self.model.parameters())
                },
                "training_params": {
                    "epochs": self.config.EPOCHS,
                    "learning_rate": self.config.LEARNING_RATE,
                    "batch_size": self.config.BATCH_SIZE,
                    "final_loss": round(final_loss, 4)
                },
                "dataset": {
                    "path": str(self.dataset_path),
                    "size_chars": dataset_size_chars
                },
                "tokenizer": {
                    "path": str(self.config.TOKENIZER_PATH),
                    "vocab_size": self.config.VOCAB_SIZE,
                }
            }

            json_path = model_path / f"{self.model_name}_config.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Metadatos guardados en: {json_path}")

    def _train_info_to_logger(self, dataset: TextDataset) -> None:
        self.logger.info(f"Iniciando entrenamiento en CPU... Total: {len(dataset)} frases de entrenamiento.")
        self.logger.info(f"El modelo tiene {sum(p.numel() for p in self.model.parameters())} parámetros.")
        self.logger.info(f"Usando dataset en: {self.dataset_path}")
        self.logger.info("Parámetros de entrenamiento:")
        self.logger.info(f" - Épocas: {self.config.EPOCHS}")
        self.logger.info(f" - Tamaño de batch: {self.config.BATCH_SIZE}")
        self.logger.info(f" - Tamaño de ventana: {self.config.MAX_SEQ_LEN}")
        self.logger.info(f" - Tamaño del vocabulario: {self.config.VOCAB_SIZE}")
        self.logger.info(f" - Tamaño del modelo: {self.config.D_MODEL}")
        self.logger.info(f" - Número de capas: {self.config.N_LAYERS}")
        self.logger.info(f" - Número de cabezas: {self.config.NUM_HEADS}")
        self.logger.info(f" - Tasa de aprendizaje: {self.config.LEARNING_RATE}")
        self.logger.info(f" - Desplazamiento: {self.stride}")

    def _save_metadata_checkpoint(self, epoch: int, loss: float) -> None:
        """
        Guarda metadatos adicionales en el checkpoint.

        Args:
            epoch(int): Época actual.
            loss(float): Pérdida actual.

        Returns:
            None
        """
        path = self.config.CHECKPOINT_DIR / self.model_name / f"{self.model_name}_checkpoint_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            'epoch': epoch,
            'loss': loss,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Metadatos del checkpoint guardados en: {path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Carga un checkpoint si existe.

        Args:
            checkpoint_path(Path): Ruta al archivo de checkpoint.
        
        Returns:
            None
        """
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Reanudando desde época {self.start_epoch}")
        else:
            self.logger.info("No se encontró checkpoint. Iniciando entrenamiento desde cero.")

    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """
        Guarda una 'foto' del estado actual del laboratorio.
        
        Args:
            epoch(int): Época actual.
            loss(float): Pérdida actual.
        
        Returns:
            None
        """
        path = self.config.CHECKPOINT_DIR / self.model_name / f"{self.model_name}_checkpoint_epoch_{epoch+1}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        self.logger.info(f"Checkpoint guardado en: {path}")

    def train(self):
        """Ejecuta el bucle principal de entrenamiento."""
        # Cargar datos
        dataset = TextDataset(str(self.dataset_path), self.tokenizer, self.config.MAX_SEQ_LEN, stride=self.stride)
        # num_workers=0 es mejor para debuggear en Windows, 
        # pin_memory ayuda a la transferencia si usaras GPU, pero en CPU no estorba.
        train_loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)

        # Sección de logger para información inicial
        self._train_info_to_logger(dataset)

        start_training_time = datetime.now()
        self.logger.info(f"Inicio del entrenamiento: {start_training_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.model.train()

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    content_size = len(f.read())

        for epoch in range(self.start_epoch, self.config.EPOCHS):
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                # Gradientes a cero
                self.optimizer.zero_grad()
                
                # Forward
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, self.config.VOCAB_SIZE), y.view(-1))
                
                # Backward y Optimización
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

                if batch_idx % 20 == 0:
                    progreso = (batch_idx / len(train_loader)) * 100
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                        f"Progreso: {progreso:.1f}% | Loss: {loss.item():.4f}"
                    )

            # Guardar checkpoint cada época (configurable)
            self.save_checkpoint(epoch, loss.item())

        # Guardado final de pesos limpios
        model_path = self.config.MODEL_DIR / self.model_name / f"{self.model_name}.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        end_time=datetime.now()
        
        # Guardar metadatos
        self._save_model_metadata(
            final_loss=loss.item(),
            dataset_size_chars=content_size, 
            model_path=model_path.parent,
            start_time=start_training_time,
            end_time=end_time
        )
        
        self.logger.info(f"Entrenamiento y documentación completados. Tiempo total: {end_time - start_training_time}")
        self.logger.info(f"Modelo final guardado en {model_path}")