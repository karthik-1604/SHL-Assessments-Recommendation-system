import sys
print(f'🐍 Python version: {sys.version}')
print('=' * 60)

def test_ml_libraries():
    print('🧪 Testing ML Libraries for RAG System...')
    
    # Test 1: Basic ML Libraries
    try:
        import numpy as np
        print('✅ NumPy:', np.__version__)
    except ImportError as e:
        print('❌ NumPy failed:', e)
        return False
    
    try:
        import pandas as pd
        print('✅ Pandas:', pd.__version__)
    except ImportError as e:
        print('❌ Pandas failed:', e)
        return False
    
    try:
        import sklearn
        print('✅ Scikit-learn:', sklearn.__version__)
    except ImportError as e:
        print('❌ Scikit-learn failed:', e)
        return False
    
    # Test 2: Embedding Models (Critical for RAG)
    print('\\n🔍 Testing Sentence Transformers...')
    try:
        from sentence_transformers import SentenceTransformer
        print('✅ Sentence Transformers import successful')
        
        # Test model loading
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print('✅ Model loading successful')
        
        # Test encoding
        test_sentences = ['Java programming test', 'Leadership assessment']
        embeddings = model.encode(test_sentences)
        print(f'✅ Encoding successful: {embeddings.shape}')
        
    except Exception as e:
        print(f'❌ Sentence Transformers failed: {e}')
        return False
    
    # Test 3: Vector Search (Critical for RAG)
    print('\\n🔍 Testing FAISS Vector Search...')
    try:
        import faiss
        print('✅ FAISS import successful')
        
        # Test index creation
        index = faiss.IndexFlatIP(384)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        print('✅ FAISS index creation successful')
        
        # Test search
        query_embedding = model.encode(['Python developer'])
        scores, indices = index.search(query_embedding.astype('float32'), k=2)
        print(f'✅ Vector search successful: Found {len(indices[0])} results')
        
    except Exception as e:
        print(f'❌ FAISS failed: {e}')
        return False
    
    # Test 4: Cross-encoder for Re-ranking (Optional but Recommended)
    print('\\n🔍 Testing Cross-Encoder for Re-ranking...')
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Test re-ranking
        pairs = [['Python developer', 'Java programming test'], 
                 ['Python developer', 'Leadership assessment']]
        scores = reranker.predict(pairs)
        print(f'✅ Cross-encoder successful: Scores {scores}')
        
    except Exception as e:
        print(f'❌ Cross-encoder failed: {e}')
        print('⚠️  Re-ranking will be limited but system can still work')
    
    print('\\n🎉 All critical ML libraries are working!')
    print('🚀 Ready to build RAG recommendation system!')
    return True

def test_simple_rag_workflow():
    print('\\n🔬 Testing Complete RAG Workflow...')
    
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Mock assessment data
        assessments = [
            'Java programming test that measures knowledge of OOP concepts',
            'Python development assessment for web frameworks',
            'Leadership skills evaluation for team management',
            'Communication assessment for client interactions'
        ]
        
        # Create embeddings
        assessment_embeddings = model.encode(assessments)
        
        # Build FAISS index
        index = faiss.IndexFlatIP(384)
        index.add(assessment_embeddings.astype('float32'))
        
        # Test query
        query = 'Need Java developer with good communication'
        query_embedding = model.encode([query])
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), k=3)
        
        print('📊 RAG Test Results:')
        for i, idx in enumerate(indices[0]):
            score = scores[0][i]
            assessment = assessments[idx]
            print(f'  {i+1}. Score: {score:.3f} | {assessment}')
        
        print('\\n✅ Complete RAG workflow successful!')
        return True
        
    except Exception as e:
        print(f'❌ RAG workflow failed: {e}')
        return False

if __name__ == '__main__':
    success = test_ml_libraries()
    if success:
        test_simple_rag_workflow()
    else:
        print('\\n❌ ML setup needs fixing before building RAG system')
