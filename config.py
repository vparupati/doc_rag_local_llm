"""
Configuration management for the RAG application.
"""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model Configuration
    model_name: str = "llama2:7b"  # Ollama model name
    model_type: str = "ollama"  # ollama or openllm
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2:7b"
    
    # OpenLLM Configuration
    openllm_model_name: str = "llama2"
    openllm_model_id: str = "meta-llama/Llama-2-7b-chat-hf"
    openllm_base_url: str = "http://localhost:3000"
    
    # Vector Store Configuration
    vector_store_path: str = "./data/vectorstore"
    chroma_persist_directory: str = "./data/chroma"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # File Upload Configuration
    max_file_size: int = 10485760  # 10MB
    allowed_extensions: List[str] = [".pdf", ".txt", ".md", ".docx"]
    
    # UI Configuration
    ui_title: str = "RAG Document Query System"
    ui_description: str = "Upload documents and query them using RAG"
    
    # Create data directories
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        Path("./uploads").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
