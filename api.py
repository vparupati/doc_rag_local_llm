"""
FastAPI application with modular endpoints for file upload and querying.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import uvicorn

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_engine import RAGEngine
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.ui_title,
    description=settings.ui_description,
    version="1.0.0"
)

# Initialize components
document_processor = DocumentProcessor()
vector_store_manager = VectorStoreManager()
rag_engine = RAGEngine(vector_store_manager)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[Dict[str, Any]]
    num_sources: int
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    llm_working: bool
    test_query_successful: bool
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    files_processed: int
    total_chunks: int
    file_details: List[Dict[str, Any]]

# Dependency to check file extension
def validate_file_extension(filename: str) -> bool:
    """Validate file extension."""
    file_ext = Path(filename).suffix.lower()
    return file_ext in settings.allowed_extensions

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "settings": settings
    })

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_info = rag_engine.health_check()
        return HealthResponse(**health_info)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            vector_store={},
            llm_working=False,
            test_query_successful=False,
            error=str(e)
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process files."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        processed_files = []
        total_chunks = 0
        
        for file in files:
            # Validate file extension
            if not validate_file_extension(file.filename):
                logger.warning(f"Skipping file with unsupported extension: {file.filename}")
                continue
            
            # Check file size
            file_content = await file.read()
            if len(file_content) > settings.max_file_size:
                logger.warning(f"File too large, skipping: {file.filename}")
                continue
            
            # Save file temporarily
            file_path = f"uploads/{file.filename}"
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            try:
                # Process document
                documents = document_processor.process_document(file_path)
                
                # Add to vector store
                doc_ids = vector_store_manager.add_documents(documents)
                
                processed_files.append({
                    "filename": file.filename,
                    "chunks": len(documents),
                    "status": "success"
                })
                
                total_chunks += len(documents)
                
                logger.info(f"Successfully processed {file.filename}: {len(documents)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                processed_files.append({
                    "filename": file.filename,
                    "chunks": 0,
                    "status": "error",
                    "error": str(e)
                })
            
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        return UploadResponse(
            message=f"Processed {len(processed_files)} files",
            files_processed=len([f for f in processed_files if f["status"] == "success"]),
            total_chunks=total_chunks,
            file_details=processed_files
        )
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process query
        response = rag_engine.query(request.question)
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar/{query}")
async def get_similar_documents(query: str, k: int = 4):
    """Get similar documents for a query."""
    try:
        documents = rag_engine.get_similar_documents(query, k=k)
        
        return {
            "query": query,
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ],
            "count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error getting similar documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def get_system_info():
    """Get system information."""
    try:
        info = rag_engine.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
async def reset_vector_store():
    """Reset the vector store."""
    try:
        vector_store_manager.reset_vector_store()
        return {"message": "Vector store reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get collection statistics."""
    try:
        stats = vector_store_manager.get_collection_info()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
