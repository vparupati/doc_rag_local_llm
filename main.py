#!/usr/bin/env python3
"""
Main application runner for the RAG system.
"""
import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from api import app
from config import settings

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "rag_app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RAG Application")
    logger.info(f"Configuration: {settings.model_name}")
    logger.info(f"API will be available at: http://{settings.api_host}:{settings.api_port}")
    
    try:
        import uvicorn
        uvicorn.run(
            "api:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
