import uvicorn
import logging
import os

from app.settings import Config, setup_logging
from app.backend.api import create_app

app = create_app(config=Config)
setup_logging()

def run_server():
    """
    Run the FastAPI server.
    """
    uvicorn.run(
        "app.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info",
        reload=False,
    )

if __name__ == "__main__":
    run_server()