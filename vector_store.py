"""
Vector store module for handling embeddings and similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations and embeddings."""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.chroma_client = None
        self._initialize_embedding_model()
        self._initialize_vector_store()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize LangChain Chroma vector store
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name="documents",
                embedding_function=self.embedding_model,
                persist_directory=settings.chroma_persist_directory
            )
            
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return []
            
            # Add documents to vector store
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores for query")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        try:
            collection = self.chroma_client.get_collection("documents")
            count = collection.count()
            
            info = {
                "total_documents": count,
                "collection_name": "documents",
                "embedding_model": settings.embedding_model,
                "persist_directory": settings.chroma_persist_directory
            }
            
            logger.info(f"Collection info: {info}")
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.chroma_client.delete_collection("documents")
            logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def reset_vector_store(self):
        """Reset the vector store (delete and recreate)."""
        try:
            self.delete_collection()
            self._initialize_vector_store()
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
            raise
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        try:
            # This is a simplified implementation
            # In practice, you might need to implement this based on your specific needs
            logger.info(f"Retrieving document with ID: {document_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            return None
