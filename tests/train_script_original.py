import torch
import logging
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from app.gpt import GPTModel, BPETokenizer, TextDataset
from app.settings import Config, setup_logging

setup_logging()
logger = logging.getLogger("TrainModule")
checkpoint_path = None

def train():
    # 1. Cargar Tokenizer y Datos
    logger.info("Cargando Tokenizer y Datos...")
    tokenizer = BPETokenizer.load(str(Config.TOKENIZER_PATH))

    logger.info("Cargando Dataset...")
    dataset = TextDataset(f"{str(Config.DATA_DIR)}/ancor.txt", tokenizer, Config.MAX_SEQ_LEN)

    logger.info("Creando DataLoader...")
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. Instanciar el Modelo
    logger.info("Instanciando el Modelo...")
    model = GPTModel(
        vocab_size=Config.VOCAB_SIZE,
        d_model=Config.D_MODEL,
        n_layers=Config.N_LAYERS,
        num_heads=Config.NUM_HEADS,
        max_seq_len=Config.MAX_SEQ_LEN
    )
    
    # 3. Optimizador y Función de Pérdida
    # AdamW es el estándar; CrossEntropy es la métrica de error para clasificación de tokens
    logger.info("Configurando el Entrenamiento...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # >>> LÓGICA DE REANUDACIÓN <<<
    if checkpoint_path is not None:
        logger.info(f"Cargando checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Cargamos los pesos al modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # CARGAMOS EL ESTADO DEL OPTIMIZADOR (Esto es lo que ocupa los 32MB extra)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Recuperamos en qué época nos quedamos
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    logger.info(f"Iniciando entrenamiento en CPU...")
    logger.info(f"Total de frases: {len(dataset)}, Batches por época: {len(train_loader)}")
    model.train() # Ponemos el modelo en modo entrenamiento

    for epoch in range(start_epoch, Config.EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            # x shape: [8, 64], y shape: [8, 64]
            
            # Forward pass
            logits = model(x) # Output: [8, 64, 5000]
            
            # Reestructuramos para CrossEntropy: (Batch * Seq, Vocab)
            loss = criterion(logits.view(-1, Config.VOCAB_SIZE), y.view(-1))
            
            # Backward pass (Cálculo de gradientes)
            optimizer.zero_grad()
            loss.backward()
            
            # Update (Ajuste de pesos)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{Config.EPOCHS+1} | Batch {batch_idx} de {len(train_loader)} ({(batch_idx/len(train_loader))*100:.2f}%) | Loss: {loss.item():.4f}")

        # Checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = Config.MODEL_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint guardado: {checkpoint_path}")


    # 4. Guardar el modelo entrenado
    torch.save(model.state_dict(), Config.MODEL_WEIGHTS_PATH)
    logger.info("Entrenamiento completado y modelo guardado.")

if __name__ == "__main__":
    train()