#!/usr/bin/env python3
"""
Example usage of the RAG application with Ollama/OpenLLM.
"""
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_manager import LLMManager
from rag_engine import RAGEngine

def main():
    """Example usage of the RAG system."""
    print("RAG System Example Usage")
    print("=" * 30)
    
    # Initialize components
    print("Initializing components...")
    document_processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager()
    llm_manager = LLMManager()
    rag_engine = RAGEngine(vector_store_manager)
    
    print("✓ All components initialized")
    
    # Check LLM health
    print(f"\nLLM Health Check: {'✓ Healthy' if llm_manager.health_check() else '✗ Unhealthy'}")
    
    # Create a sample document
    print("\nCreating sample document...")
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can perform tasks that typically require human intelligence. 
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.
    
    Machine Learning is a subset of AI that focuses on the development of algorithms and 
    statistical models that enable computer systems to improve their performance on a specific 
    task through experience, without being explicitly programmed.
    
    Deep Learning is a subset of machine learning that uses artificial neural networks with 
    multiple layers to model and understand complex patterns in data.
    """
    
    sample_file = Path("sample_ai_document.txt")
    sample_file.write_text(sample_text)
    
    try:
        # Process the document
        print("Processing document...")
        documents = document_processor.process_document(str(sample_file))
        print(f"✓ Document processed into {len(documents)} chunks")
        
        # Add to vector store
        print("Adding to vector store...")
        doc_ids = vector_store_manager.add_documents(documents)
        print(f"✓ Added {len(doc_ids)} document chunks to vector store")
        
        # Example queries
        queries = [
            "What is artificial intelligence?",
            "What is the difference between machine learning and deep learning?",
            "What are neural networks used for?"
        ]
        
        print("\nRunning example queries...")
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 50)
            
            try:
                response = rag_engine.query(query)
                print(f"Answer: {response['answer']}")
                print(f"Sources: {response['num_sources']} documents")
            except Exception as e:
                print(f"Error: {e}")
        
        # Get system info
        print("\nSystem Information:")
        print("-" * 20)
        info = rag_engine.get_model_info()
        print(f"Model Type: {info.get('model_type', 'Unknown')}")
        print(f"Model Name: {info.get('model_name', 'Unknown')}")
        print(f"Vector Store: {info.get('vector_store_info', {}).get('total_documents', 0)} documents")
        
    except Exception as e:
        print(f"Error during example: {e}")
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()
        print("\n✓ Cleanup completed")

if __name__ == "__main__":
    main()
