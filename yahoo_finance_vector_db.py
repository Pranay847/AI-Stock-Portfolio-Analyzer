"""
Simple Yahoo Finance to Vector Database
Fetch stock data → Create embeddings → Store in ChromaDB
"""

import yfinance as yf
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json

class StockVectorDB:
    def __init__(self, db_path="./stock_db"):
        """Initialize vector database and embedding model"""
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create collection for stocks
        self.collection = self.client.get_or_create_collection(
            name="stocks",
            metadata={"description": "Stock market data"}
        )
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Vector database initialized")
    
    def fetch_stock_data(self, symbol):
        """Fetch real-time stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get current price
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            # Prepare stock data
            data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'price': price,
                'previous_close': info.get('previousClose', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'volume': info.get('volume', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'description': info.get('longBusinessSummary', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Fetched {symbol}: ${price}")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")
            return None
    
    def create_embedding(self, stock_data):
        """Convert stock data to text and create embedding"""
        # Create descriptive text from stock data
        text = f"""
        {stock_data['name']} ({stock_data['symbol']})
        Sector: {stock_data['sector']}
        Industry: {stock_data['industry']}
        Price: ${stock_data['price']}
        Market Cap: ${stock_data['market_cap']}
        P/E Ratio: {stock_data['pe_ratio']}
        Description: {stock_data['description'][:300]}
        """
        
        # Generate embedding vector
        embedding = self.model.encode(text).tolist()
        
        return text, embedding
    
    def store_in_vectordb(self, stock_data):
        """Store stock data with embedding in vector database"""
        if not stock_data:
            return False
        
        try:
            # Create embedding
            text, embedding = self.create_embedding(stock_data)
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=[stock_data['symbol']],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    'symbol': stock_data['symbol'],
                    'name': stock_data['name'],
                    'price': stock_data['price'],
                    'sector': stock_data['sector'],
                    'industry': stock_data['industry'],
                    'market_cap': stock_data['market_cap'],
                    'pe_ratio': stock_data['pe_ratio'],
                    'timestamp': stock_data['timestamp'],
                    'full_data': json.dumps(stock_data)
                }]
            )
            
            print(f"✅ Stored {stock_data['symbol']} in vector database")
            return True
            
        except Exception as e:
            print(f"❌ Error storing data: {str(e)}")
            return False
    
    def process_stock(self, symbol):
        """Complete pipeline: Fetch → Embed → Store"""
        print(f"\n📊 Processing {symbol}...")
        
        # Step 1: Fetch from Yahoo Finance
        stock_data = self.fetch_stock_data(symbol)
        
        if stock_data:
            # Step 2 & 3: Create embedding and store
            self.store_in_vectordb(stock_data)
        
        return stock_data
    
    def process_multiple_stocks(self, symbols):
        """Process multiple stocks"""
        results = []
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}]")
            result = self.process_stock(symbol)
            results.append(result)
        
        print(f"\n✅ Completed! Processed {len(symbols)} stocks")
        return results
    
    def search_stocks(self, query, n_results=5):
        """Search stocks using natural language query"""
        # Create embedding for query
        query_embedding = self.model.encode(query).tolist()
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        stocks = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            stocks.append({
                'symbol': metadata['symbol'],
                'name': metadata['name'],
                'price': metadata['price'],
                'sector': metadata['sector'],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return stocks
    
    def get_stock(self, symbol):
        """Retrieve specific stock from database"""
        try:
            result = self.collection.get(
                ids=[symbol],
                include=['metadatas']
            )
            
            if result['ids']:
                metadata = result['metadatas'][0]
                return json.loads(metadata['full_data'])
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving {symbol}: {str(e)}")
            return None
    
    def get_stats(self):
        """Get database statistics"""
        count = self.collection.count()
        return {
            'total_stocks': count
        }


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    
    # Initialize
    db = StockVectorDB(db_path="./my_stocks")
    
    print("\n" + "="*60)
    print("YAHOO FINANCE → VECTOR DATABASE")
    print("="*60)
    
    # ===== EXAMPLE 1: Single Stock =====
    print("\n📌 EXAMPLE 1: Process Single Stock")
    db.process_stock("AAPL")
    
    # ===== EXAMPLE 2: Multiple Stocks =====
    print("\n📌 EXAMPLE 2: Process Multiple Stocks")
    portfolio = ["MSFT", "GOOGL", "TSLA", "NVDA"]
    db.process_multiple_stocks(portfolio)
    
    # ===== EXAMPLE 3: Search with Natural Language =====
    print("\n📌 EXAMPLE 3: Semantic Search")
    results = db.search_stocks("technology companies", n_results=3)
    
    print("\nSearch Results:")
    for stock in results:
        print(f"  • {stock['symbol']}: {stock['name']} - ${stock['price']:.2f}")
    
    # ===== EXAMPLE 4: Retrieve Specific Stock =====
    print("\n📌 EXAMPLE 4: Get Specific Stock")
    aapl = db.get_stock("AAPL")
    if aapl:
        print(f"  Symbol: {aapl['symbol']}")
        print(f"  Name: {aapl['name']}")
        print(f"  Price: ${aapl['price']:.2f}")
        print(f"  Sector: {aapl['sector']}")
    
    # ===== EXAMPLE 5: Database Stats =====
    print("\n📌 EXAMPLE 5: Database Statistics")
    stats = db.get_stats()
    print(f"  Total stocks stored: {stats['total_stocks']}")
    
    print("\n" + "="*60)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("="*60 + "\n")

    
   
