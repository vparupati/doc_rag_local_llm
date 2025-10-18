#!/usr/bin/env python3
"""
Setup script for Ollama and OpenLLM models.
"""
import subprocess
import sys
import logging
import requests
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama."""
    logger.info("Installing Ollama...")
    try:
        # Download and install Ollama
        subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], 
                      stdout=subprocess.PIPE, text=True)
        subprocess.run(['sh', '-'], input='curl -fsSL https://ollama.ai/install.sh | sh', 
                      shell=True, text=True)
        logger.info("Ollama installed successfully")
        return True
    except Exception as e:
        logger.error(f"Error installing Ollama: {str(e)}")
        return False

def start_ollama():
    """Start Ollama service."""
    logger.info("Starting Ollama service...")
    try:
        # Start Ollama in background
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # Wait for service to start
        logger.info("Ollama service started")
        return True
    except Exception as e:
        logger.error(f"Error starting Ollama: {str(e)}")
        return False

def check_ollama_running():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def pull_ollama_model(model_name="llama2:7b"):
    """Pull a model from Ollama."""
    logger.info(f"Pulling Ollama model: {model_name}")
    try:
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Successfully pulled {model_name}")
            return True
        else:
            logger.error(f"Error pulling model: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error pulling model: {str(e)}")
        return False

def setup_openllm():
    """Setup OpenLLM."""
    logger.info("Setting up OpenLLM...")
    try:
        # Install OpenLLM
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'openllm'], check=True)
        
        # Start OpenLLM server
        logger.info("Starting OpenLLM server...")
        subprocess.Popen([
            sys.executable, '-m', 'openllm', 'start', 'llama2',
            '--port', '3000',
            '--host', '0.0.0.0'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        time.sleep(30)
        logger.info("OpenLLM server started")
        return True
    except Exception as e:
        logger.error(f"Error setting up OpenLLM: {str(e)}")
        return False

def check_openllm_running():
    """Check if OpenLLM is running."""
    try:
        response = requests.get("http://localhost:3000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main setup function."""
    print("LLM Setup for RAG Application")
    print("=" * 40)
    
    # Check what's already installed
    ollama_installed = check_ollama_installed()
    ollama_running = check_ollama_running()
    openllm_running = check_openllm_running()
    
    print(f"Ollama installed: {ollama_installed}")
    print(f"Ollama running: {ollama_running}")
    print(f"OpenLLM running: {openllm_running}")
    
    # Setup Ollama
    if not ollama_installed:
        print("\nSetting up Ollama...")
        if install_ollama():
            ollama_installed = True
        else:
            print("Failed to install Ollama")
            return 1
    
    if not ollama_running and ollama_installed:
        print("\nStarting Ollama...")
        if start_ollama():
            time.sleep(5)
            if check_ollama_running():
                print("Ollama is running")
            else:
                print("Failed to start Ollama")
                return 1
        else:
            print("Failed to start Ollama")
            return 1
    
    if ollama_running or check_ollama_running():
        print("\nPulling Llama2 model...")
        if pull_ollama_model("llama2:7b"):
            print("Llama2 model ready")
        else:
            print("Failed to pull Llama2 model")
    
    # Setup OpenLLM (optional)
    print("\nSetting up OpenLLM (optional)...")
    if not openllm_running:
        if setup_openllm():
            if check_openllm_running():
                print("OpenLLM is running")
            else:
                print("OpenLLM setup completed but not running")
        else:
            print("OpenLLM setup failed")
    
    print("\nSetup completed!")
    print("\nTo use Ollama:")
    print("  ollama serve")
    print("  ollama pull llama2:7b")
    print("\nTo use OpenLLM:")
    print("  openllm start llama2 --port 3000")
    print("\nTo start the RAG application:")
    print("  python main.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
