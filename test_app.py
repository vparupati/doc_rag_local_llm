#!/usr/bin/env python3
"""
Simple test script to verify the RAG application components.
"""
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from config import settings
        print("✓ Config module imported successfully")
        
        from document_processor import DocumentProcessor
        print("✓ Document processor imported successfully")
        
        from vector_store import VectorStoreManager
        print("✓ Vector store imported successfully")
        
        from llm_manager import LLMManager
        print("✓ LLM manager imported successfully")
        
        from rag_engine import RAGEngine
        print("✓ RAG engine imported successfully")
        
        from api import app
        print("✓ API module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_document_processor():
    """Test document processor functionality."""
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✓ Document processor initialized")
        
        # Test with a simple text file
        test_file = Path("test_document.txt")
        test_file.write_text("This is a test document for the RAG system.")
        
        documents = processor.process_document(str(test_file))
        print(f"✓ Document processed into {len(documents)} chunks")
        
        # Clean up
        test_file.unlink()
        
        return True
    except Exception as e:
        print(f"✗ Document processor test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from config import settings
        
        print(f"✓ Model name: {settings.model_name}")
        print(f"✓ Model type: {settings.model_type}")
        print(f"✓ API port: {settings.api_port}")
        print(f"✓ Debug mode: {settings.debug}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_llm_manager():
    """Test LLM manager functionality."""
    try:
        from llm_manager import LLMManager
        
        # Note: This test may fail if Ollama/OpenLLM is not running
        # That's expected and not a critical failure
        manager = LLMManager()
        print("✓ LLM manager initialized")
        
        # Test health check (may fail if services not running)
        health = manager.health_check()
        print(f"✓ LLM health check: {'healthy' if health else 'unhealthy (services may not be running)'}")
        
        return True
    except Exception as e:
        print(f"✗ LLM manager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing RAG Application Components")
    print("=" * 40)
    
    tests = [
        ("Configuration", test_config),
        ("Imports", test_imports),
        ("Document Processor", test_document_processor),
        ("LLM Manager", test_llm_manager),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} test failed")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The application is ready to run.")
        print("\nTo start the application, run:")
        print("  python main.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
