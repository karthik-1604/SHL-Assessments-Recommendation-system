#!/usr/bin/env python3
"""
Enhanced ML libraries test for SHL Assessment Recommendation System
"""

import sys
import importlib.util

def test_import(module_name, package_name=None):
    """Test if a module can be imported successfully"""
    try:
        if package_name:
            module = __import__(package_name, fromlist=[module_name])
            getattr(module, module_name)
        else:
            __import__(module_name)
        return True, None
    except Exception as e:
        return False, str(e)

def get_version(module_name):
    """Get version of an imported module"""
    try:
        module = __import__(module_name)
        return getattr(module, '__version__', 'Unknown')
    except:
        return 'Not installed'

def main():
    print(f"🐍 Python version: {sys.version}")
    print("=" * 60)
    print("🧪 Testing ML Libraries for RAG System...")
    
    # Core libraries
    core_libs = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'), 
        ('sklearn', 'Scikit-learn'),
    ]
    
    for lib, name in core_libs:
        success, error = test_import(lib)
        if success:
            version = get_version(lib)
            print(f"✅ {name}: {version}")
        else:
            print(f"❌ {name} failed: {error}")
    
    # Sentence Transformers with detailed error checking
    print("\n🔍 Testing Sentence Transformers...")
    
    # First test huggingface_hub
    success, error = test_import('huggingface_hub')
    if success:
        hf_version = get_version('huggingface_hub')
        print(f"✅ Huggingface Hub: {hf_version}")
        
        # Test HF_HUB_CACHE specifically
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            print("✅ HF_HUB_CACHE is accessible")
        except Exception as e:
            print(f"❌ HF_HUB_CACHE error: {e}")
            
    else:
        print(f"❌ Huggingface Hub failed: {error}")
    
    # Test sentence transformers
    success, error = test_import('sentence_transformers')
    if success:
        st_version = get_version('sentence_transformers')
        print(f"✅ Sentence Transformers: {st_version}")
        
        # Test actual model loading
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            test_embedding = model.encode(["test sentence"])
            print(f"✅ Model loading successful - embedding shape: {test_embedding.shape}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
    else:
        print(f"❌ Sentence Transformers failed: {error}")
    
    # Additional ML libraries for RAG
    print("\n🔧 Testing Additional Libraries...")
    
    additional_libs = [
        ('faiss', 'FAISS'),
        ('chromadb', 'ChromaDB'),
        ('openai', 'OpenAI'),
        ('langchain', 'LangChain'),
        ('nltk', 'NLTK'),
    ]
    
    for lib, name in additional_libs:
        success, error = test_import(lib)
        if success:
            version = get_version(lib)
            print(f"✅ {name}: {version}")
        else:
            print(f"❌ {name} failed: {error}")
    
    # Test PyTorch
    print("\n🔥 Testing PyTorch...")
    success, error = test_import('torch')
    if success:
        torch_version = get_version('torch')
        print(f"✅ PyTorch: {torch_version}")
        
        try:
            import torch
            print(f"✅ CUDA available: {torch.cuda.is_available()}")
        except Exception as e:
            print(f"❌ PyTorch CUDA check failed: {e}")
    else:
        print(f"❌ PyTorch failed: {error}")
    
    print("\n" + "=" * 60)
    
    # Overall assessment
    critical_libs = ['numpy', 'pandas', 'sklearn', 'sentence_transformers', 'torch']
    failed_critical = []
    
    for lib in critical_libs:
        success, _ = test_import(lib)
        if not success:
            failed_critical.append(lib)
    
    if failed_critical:
        print(f"❌ Critical libraries failed: {', '.join(failed_critical)}")
        print("🔧 Run the fix commands to resolve issues")
        return False
    else:
        print("✅ All critical ML libraries are working!")
        print("🚀 Ready to build the RAG system!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)