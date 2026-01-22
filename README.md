# Cachito-GPT

**Cachito-GPT** es un ecosistema completo para el desarrollo, entrenamiento y despliegue de modelos de lenguaje autorregresivos (Transformers) basados en la arquitectura GPT, desarrollado íntegramente en Python y PyTorch.

Este proyecto nace como parte los trabajos en el marco de la clase de Microprocesadores del Sexto Semestre de Telecomunicaciones, en la UNEFA Extensión Bejuma.

## 🚀 Características Principales

* **Arquitectura Modular**: Implementación desde cero de bloques de *Multi-Head Attention*, *Positional Encoding* y *Feed-Forward Networks*.
* **Pipeline de Entrenamiento Eficiente**: Clase `TrainModule` con soporte para checkpoints automáticos, exportación de metadatos en JSON y optimización de datos mediante `stride` dinámico.
* **Tokenizer BPE Local**: Implementación de *Byte Pair Encoding* personalizada para el manejo eficiente del vocabulario.
* **Inferencia Avanzada**: Motor de generación con filtros de muestreo *Top-K*, *Top-P*, temperatura y penalización por repetición.
* **API & UI**: Backend robusto con FastAPI y una interfaz web moderna construida con Bulma CSS y Jinja2.

## 📂 Estructura del Proyecto

```text
app/
├── backend/          # API REST (FastAPI) y Servicios
├── gpt/              # Core del modelo (Arquitectura, Dataset, Tokenizer)
├── ui/               # Interfaz Gráfica (HTML, CSS, JS)
├── settings/         # Configuración centralizada y logging
├── trainer.py        # Lógica de entrenamiento OO
├── main.py           # Entrypoint para ejecución de la API
└── generator.py      # Motor de inferencia
data/                 # Corpus de texto (.txt)
models/               # Pesos (.pth) y Metadatos (.json)
tokenizer/            # Archivos del Tokenizer entrenado

```

## 🛠️ Requisitos Técnicos

Como ingeniero, este proyecto está diseñado para ejecutarse en entornos locales, optimizado incluso para hardware modesto (probado en CPU Intel i3).

* **Lenguaje**: Python 3.10+
* **Deep Learning**: PyTorch
* **Web Stack**: FastAPI, Uvicorn, Jinja2, Pydantic
* **Frontend**: Bulma CSS, FontAwesome

## 🚀 Instalación y Uso

1. **Clonar y preparar entorno**:
```bash
python -m venv venv
source venv/scripts/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

```


2. **Entrenamiento**:
Configura tus parámetros en `app/settings/config.py` y ejecuta el entrenamiento.
```bash
python -m app.trainer

```


3. **Ejecutar la API y Web UI**:
```bash
python -m app.main

```
Por defecto, la api ejecuta cachito_2, si has entrenado un modelo diferente, debes ingresar a `app/backden/services/cachito_service.py` y modificar el generator en el constructor de la clase para que cargue el modelo que has creado. De momento no se ha centralizado ni se ha incluido un selector de modelos en la interfaz.

Accede a la interfaz en `http://localhost:8000`.

## 🛡️ Filosofía Open Source
---

Desarrollado con fines educativos por el 6to semestre de Ingeniería en Telecomunicaciones, UNAFA Bejuma.