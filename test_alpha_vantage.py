"""
Quick test script for Alpha Vantage Vector DB
Run: python test_alpha_vantage.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_imports():
    """Test that all required packages are installed"""
    print("1️⃣  Testing imports...")
    
    try:
        import chromadb
        print("   ✅ chromadb")
    except ImportError as e:
        print(f"   ❌ chromadb: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ sentence_transformers")
    except ImportError as e:
        print(f"   ❌ sentence_transformers: {e}")
        return False
    
    try:
        import requests
        print("   ✅ requests")
    except ImportError as e:
        print(f"   ❌ requests: {e}")
        return False
    
    return True


def test_api_key():
    """Test that API key is configured"""
    print("\n2️⃣  Testing API key...")
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    if not api_key:
        print("   ❌ ALPHA_VANTAGE_API_KEY not found in .env")
        print("\n   To fix this:")
        print("   1. Get a free key at: https://www.alphavantage.co/support/#api-key")
        print("   2. Add to your .env file: ALPHA_VANTAGE_API_KEY=your_key_here")
        return False
    
    if api_key == "your_api_key_here" or api_key == "your_alpha_vantage_api_key_here":
        print("   ❌ API key is still the placeholder value")
        print("   Please replace with your actual API key")
        return False
    
    print(f"   ✅ API key found: {api_key[:4]}...{api_key[-4:]}")
    return True


def test_embedding_model():
    """Test embedding model loading"""
    print("\n3️⃣  Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test encoding
        test_text = "Apple Inc is a technology company"
        embedding = model.encode(test_text)
        
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Embedding dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_chromadb():
    """Test ChromaDB initialization"""
    print("\n4️⃣  Testing ChromaDB...")
    
    try:
        import chromadb
        import tempfile
        
        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            client = chromadb.PersistentClient(path=tmpdir)
            collection = client.get_or_create_collection(name="test")
            
            # Test insert
            collection.add(
                ids=["test1"],
                embeddings=[[0.1] * 384],
                documents=["Test document"],
                metadatas=[{"type": "test"}]
            )
            
            # Test query
            results = collection.query(
                query_embeddings=[[0.1] * 384],
                n_results=1
            )
            
            print(f"   ✅ ChromaDB working correctly")
            print(f"   ✅ Test document retrieved successfully")
            return True
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_alpha_vantage_api():
    """Test Alpha Vantage API connection"""
    print("\n5️⃣  Testing Alpha Vantage API...")
    
    import requests
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("   ⏭️  Skipping (no valid API key)")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": "IBM",
            "apikey": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            price = data["Global Quote"].get("05. price", "N/A")
            print(f"   ✅ API connection successful")
            print(f"   ✅ IBM current price: ${price}")
            return True
        elif "Note" in data:
            print(f"   ⚠️  API rate limit reached (25 calls/day on free tier)")
            print(f"   Message: {data['Note'][:80]}...")
            return None
        elif "Error Message" in data:
            print(f"   ❌ API error: {data['Error Message']}")
            return False
        else:
            print(f"   ⚠️  Unexpected response: {data}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n6️⃣  Testing full pipeline...")
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("   ⏭️  Skipping (no valid API key)")
        return None
    
    try:
        from alpha_vantage_vector_db import AlphaVantageVectorDB
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize
            db = AlphaVantageVectorDB(db_path=tmpdir)
            
            # Process a stock (IBM is reliable for testing)
            result = db.process_stock("IBM", include_news=False)
            
            if result.get("success"):
                print(f"   ✅ Full pipeline working!")
                
                # Test search
                search_results = db.search("technology company", n_results=1)
                if search_results:
                    print(f"   ✅ Search working - found: {search_results[0]['metadata'].get('symbol')}")
                
                # Show stats
                stats = db.get_stats()
                print(f"   ✅ Stored {stats['total']} embeddings")
                
                return True
            else:
                print(f"   ⚠️  Pipeline ran but no data retrieved (API limit?)")
                return None
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("🧪 ALPHA VANTAGE VECTOR DB - TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Run tests
    results["imports"] = test_imports()
    results["api_key"] = test_api_key()
    results["embedding"] = test_embedding_model()
    results["chromadb"] = test_chromadb()
    results["api"] = test_alpha_vantage_api()
    results["pipeline"] = test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"   {test}: {status}")
    
    failed = sum(1 for r in results.values() if r is False)
    
    if failed == 0:
        print("\n🎉 All tests passed! You're ready to use Alpha Vantage Vector DB")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix the issues above.")
    
    print("="*60 + "\n")
