# RAG Document Query System

A modular Python application for building Retrieval-Augmented Generation (RAG) systems with file processing, vector storage, and querying capabilities using Ollama/OpenLLM models.

## Features

- **Document Processing**: Support for PDF, TXT, MD, and DOCX files
- **Vector Storage**: ChromaDB integration for efficient similarity search
- **LLM Integration**: Support for Ollama and OpenLLM models
- **Web UI**: Modern, responsive interface for file upload and querying
- **Modular Architecture**: Easy to debug and extend
- **RESTful API**: Complete API for integration with other systems

## Project Structure

```
├── api.py                 # FastAPI application with endpoints
├── config.py              # Configuration management
├── document_processor.py  # Document processing and text extraction
├── vector_store.py        # Vector store and embedding management
├── rag_engine.py          # RAG query engine with Ollama/OpenLLM integration
├── llm_manager.py         # LLM provider management (Ollama/OpenLLM)
├── setup_llm.py          # Setup script for LLM services
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   └── index.html
├── static/               # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── data/                 # Data storage directories
│   ├── vectorstore/
│   └── chroma/
├── uploads/              # Temporary file uploads
└── logs/                 # Application logs
```

## Installation

1. **Clone or download the project**:
   ```bash
   cd /path/to/your/project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up LLM services** (Ollama or OpenLLM):
   ```bash
   python setup_llm.py
   ```

5. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

## Usage

### Starting the Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

### Using the Web Interface

1. **Upload Documents**: Use the upload section to add PDF, TXT, MD, or DOCX files
2. **Query Documents**: Ask questions about your uploaded documents
3. **View Statistics**: Check system status and document counts
4. **Reset System**: Clear all uploaded documents if needed

### Using the API

#### Upload Files
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@document1.pdf" \
     -F "files=@document2.txt"
```

#### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
```

#### Get System Health
```bash
curl "http://localhost:8000/health"
```

#### Get Statistics
```bash
curl "http://localhost:8000/stats"
```

## Configuration

The application can be configured through the `config.py` file or environment variables:

- `MODEL_NAME`: LLM model to use (default: llama2:7b for Ollama)
- `MODEL_TYPE`: LLM provider type (ollama or openllm)
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Ollama model name (default: llama2:7b)
- `OPENLLM_BASE_URL`: OpenLLM server URL (default: http://localhost:3000)
- `OPENLLM_MODEL_NAME`: OpenLLM model name (default: llama2)
- `EMBEDDING_MODEL`: Embedding model for vectorization
- `API_HOST`: API host address (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 10MB)
- `ALLOWED_EXTENSIONS`: Supported file extensions

## Model Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: Optional but recommended for better performance
- **Storage**: At least 5GB free space for models and data

### Model Setup
The application supports two LLM providers:

#### Ollama (Recommended)
- Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
- Start Ollama: `ollama serve`
- Pull a model: `ollama pull llama2:7b`
- Models are automatically managed by Ollama

#### OpenLLM
- Install OpenLLM: `pip install openllm`
- Start server: `openllm start llama2 --port 3000`
- Models are automatically managed by OpenLLM

Both providers will download models on first use. Make sure you have:
- Sufficient disk space for model files (5-20GB depending on model)
- Stable internet connection for initial download

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure you have sufficient RAM/VRAM
   - Check internet connection for model downloads
   - Verify Hugging Face access permissions

2. **File Upload Issues**:
   - Check file size limits
   - Verify file format is supported
   - Ensure sufficient disk space

3. **Memory Issues**:
   - Reduce chunk size in document processor
   - Use smaller embedding models
   - Enable model quantization

### Debug Mode

Run with debug mode enabled for detailed logging:
```bash
DEBUG=True python main.py
```

### Logs

Check application logs in the `logs/` directory:
```bash
tail -f logs/rag_app.log
```

## Development

### Adding New File Formats

1. Add extraction method in `document_processor.py`
2. Update `extract_text()` method
3. Add file extension to `allowed_extensions` in config

### Customizing the UI

- Modify templates in `templates/`
- Update styles in `static/css/`
- Add JavaScript functionality in `static/js/`

### Extending the API

- Add new endpoints in `api.py`
- Update Pydantic models for request/response
- Add corresponding UI components

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Create an issue with detailed information
