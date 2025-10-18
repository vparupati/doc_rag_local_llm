"""
RAG (Retrieval-Augmented Generation) engine with Ollama/OpenLLM integration.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms.base import LLM

from vector_store import VectorStoreManager
from llm_manager import LLMManager
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLLM(LLM):
    """Custom LangChain LLM wrapper for our LLM manager."""
    
    llm_manager: LLMManager
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__(llm_manager=llm_manager)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM with the given prompt."""
        try:
            return self.llm_manager.generate(prompt)
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type."""
        return "custom_llm"

class RAGEngine:
    """RAG engine for document querying with Ollama/OpenLLM models."""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm_manager = None
        self.llm = None
        self.qa_chain = None
        self._initialize_llm_manager()
        self._initialize_llm()
        self._initialize_qa_chain()
    
    def _initialize_llm_manager(self):
        """Initialize the LLM manager."""
        try:
            self.llm_manager = LLMManager()
            logger.info(f"LLM manager initialized with {settings.model_type}")
        except Exception as e:
            logger.error(f"Error initializing LLM manager: {str(e)}")
            raise
    
    def _initialize_llm(self):
        """Initialize the LangChain LLM wrapper."""
        try:
            self.llm = CustomLLM(self.llm_manager)
            logger.info("LangChain LLM wrapper initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM wrapper: {str(e)}")
            raise
    
    def _initialize_qa_chain(self):
        """Initialize the question-answering chain."""
        try:
            # Define custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question: {question}

            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store_manager.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info("QA chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            logger.info(f"Processing query: {question}")
            
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Extract relevant information
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc in result["source_documents"]
                ],
                "num_sources": len(result["source_documents"])
            }
            
            logger.info(f"Query processed successfully. Found {response['num_sources']} sources.")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "num_sources": 0,
                "error": str(e)
            }
    
    def get_similar_documents(self, query: str, k: int = 4) -> List[Document]:
        """Get similar documents for a query."""
        try:
            return self.vector_store_manager.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error getting similar documents: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            llm_info = self.llm_manager.get_model_info() if self.llm_manager else {}
            info = {
                "model_name": settings.model_name,
                "model_type": settings.model_type,
                "embedding_model": settings.embedding_model,
                "llm_provider_info": llm_info,
                "vector_store_info": self.vector_store_manager.get_collection_info()
            }
            return info
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the RAG system."""
        try:
            # Test vector store
            vector_store_info = self.vector_store_manager.get_collection_info()
            
            # Test LLM provider
            llm_healthy = self.llm_manager.health_check() if self.llm_manager else False
            
            # Test LLM with a simple query if healthy
            test_query_successful = False
            if llm_healthy:
                try:
                    test_response = self.query("What is this system about?")
                    test_query_successful = "error" not in test_response
                except Exception as e:
                    logger.warning(f"Test query failed: {str(e)}")
            
            health_status = {
                "status": "healthy" if llm_healthy and test_query_successful else "unhealthy",
                "vector_store": vector_store_info,
                "llm_working": llm_healthy,
                "test_query_successful": test_query_successful,
                "llm_provider": settings.model_type
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
