"""
LLM Manager for Ollama and OpenLLM integration.
"""
import logging
import requests
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider is healthy."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass

class OllamaProvider(LLMProvider):
    """Ollama LLM provider."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Ollama client."""
        try:
            import ollama
            self.client = ollama.Client(host=self.base_url)
            logger.info(f"Ollama client initialized with base URL: {self.base_url}")
        except Exception as e:
            logger.error(f"Error initializing Ollama client: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                **kwargs
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                current_model = next(
                    (m for m in models if m['name'] == self.model), 
                    None
                )
                return {
                    "provider": "ollama",
                    "model": self.model,
                    "base_url": self.base_url,
                    "available_models": [m['name'] for m in models],
                    "current_model_info": current_model
                }
            return {"provider": "ollama", "error": "Failed to get model info"}
        except Exception as e:
            return {"provider": "ollama", "error": str(e)}

class OpenLLMProvider(LLMProvider):
    """OpenLLM LLM provider."""
    
    def __init__(self, base_url: str = None, model_name: str = None):
        self.base_url = base_url or settings.openllm_base_url
        self.model_name = model_name or settings.openllm_model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenLLM client."""
        try:
            import openllm
            self.client = openllm.client.HTTPClient(self.base_url)
            logger.info(f"OpenLLM client initialized with base URL: {self.base_url}")
        except Exception as e:
            logger.error(f"Error initializing OpenLLM client: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenLLM."""
        try:
            response = self.client.generate(
                prompt=prompt,
                **kwargs
            )
            return response.outputs[0].text
        except Exception as e:
            logger.error(f"Error generating text with OpenLLM: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """Check if OpenLLM is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenLLM health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenLLM model information."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get('data', [])
                return {
                    "provider": "openllm",
                    "model_name": self.model_name,
                    "base_url": self.base_url,
                    "available_models": [m['id'] for m in models],
                    "models_info": models
                }
            return {"provider": "openllm", "error": "Failed to get model info"}
        except Exception as e:
            return {"provider": "openllm", "error": str(e)}

class LLMManager:
    """Manager for LLM providers."""
    
    def __init__(self):
        self.provider = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the appropriate LLM provider."""
        try:
            if settings.model_type.lower() == "ollama":
                self.provider = OllamaProvider()
                logger.info("Initialized Ollama provider")
            elif settings.model_type.lower() == "openllm":
                self.provider = OpenLLMProvider()
                logger.info("Initialized OpenLLM provider")
            else:
                raise ValueError(f"Unsupported model type: {settings.model_type}")
        except Exception as e:
            logger.error(f"Error initializing LLM provider: {str(e)}")
            # Fallback to Ollama if available
            try:
                self.provider = OllamaProvider()
                logger.info("Fallback to Ollama provider")
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {str(fallback_error)}")
                raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the current provider."""
        if not self.provider:
            raise RuntimeError("No LLM provider initialized")
        return self.provider.generate(prompt, **kwargs)
    
    def health_check(self) -> bool:
        """Check if the current provider is healthy."""
        if not self.provider:
            return False
        return self.provider.health_check()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.provider:
            return {"error": "No provider initialized"}
        return self.provider.get_model_info()
    
    def switch_provider(self, provider_type: str):
        """Switch to a different provider."""
        try:
            if provider_type.lower() == "ollama":
                self.provider = OllamaProvider()
            elif provider_type.lower() == "openllm":
                self.provider = OpenLLMProvider()
            else:
                raise ValueError(f"Unsupported provider type: {provider_type}")
            logger.info(f"Switched to {provider_type} provider")
        except Exception as e:
            logger.error(f"Error switching provider: {str(e)}")
            raise
