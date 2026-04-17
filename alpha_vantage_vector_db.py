import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def clean_symbol(symbol: str) -> str:
    """Clean and validate stock symbol."""
    if not symbol:
        return ""
    # Strip whitespace and convert to uppercase
    cleaned = symbol.strip().upper()
    # Remove any non-alphanumeric characters except hyphen (for symbols like BRK-B)
    cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '-')
    return cleaned


class AlphaVantageVectorDB:
    """
    Complete pipeline for fetching stock data from Alpha Vantage,
    creating embeddings, and storing in a vector database.
    """
    
    def __init__(self, db_path: str = "./alpha_vantage_vector_db"):
        """
        Initialize vector database and embedding model.
        
        Args:
            db_path: Path to store the ChromaDB database
        """
        # Get API key from environment
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY not found in environment. "
                "Get a free key at https://www.alphavantage.co/support/#api-key"
            )
        
        self.base_url = "https://www.alphavantage.co/query"
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create collections for different data types
        self.quotes_collection = self.client.get_or_create_collection(
            name="stock_quotes",
            metadata={"description": "Real-time stock quotes from Alpha Vantage"}
        )
        
        self.overview_collection = self.client.get_or_create_collection(
            name="company_overviews",
            metadata={"description": "Company fundamental data"}
        )
        
        self.news_collection = self.client.get_or_create_collection(
            name="stock_news",
            metadata={"description": "Stock news and sentiment"}
        )
        
        # Load embedding model (all-MiniLM-L6-v2 is fast and good quality)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Alpha Vantage Vector DB initialized")
        print(f"   📁 Database path: {db_path}")
    
    # ==================== DATA FETCHING ====================
    
    def fetch_quote(self, symbol: str) -> Optional[dict]:
        """
        Fetch real-time quote data for a stock.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            Dictionary with quote data or None if failed
        """
        symbol = clean_symbol(symbol)
        if not symbol:
            print(f"❌ Invalid symbol provided")
            return None
        
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Global Quote" not in data or not data["Global Quote"]:
                print(f"❌ No quote data for {symbol}")
                return None
            
            quote = data["Global Quote"]
            
            # Parse and clean the data
            result = {
                "symbol": quote.get("01. symbol", symbol),
                "price": float(quote.get("05. price", 0)),
                "open": float(quote.get("02. open", 0)),
                "high": float(quote.get("03. high", 0)),
                "low": float(quote.get("04. low", 0)),
                "volume": int(quote.get("06. volume", 0)),
                "previous_close": float(quote.get("08. previous close", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": quote.get("10. change percent", "0%").replace("%", ""),
                "latest_trading_day": quote.get("07. latest trading day", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Fetched quote for {symbol}: ${result['price']:.2f}")
            return result
            
        except Exception as e:
            print(f"❌ Error fetching quote for {symbol}: {str(e)}")
            return None
    
    def fetch_company_overview(self, symbol: str) -> Optional[dict]:
        """
        Fetch company fundamental data (overview).
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company data or None if failed
        """
        symbol = clean_symbol(symbol)
        if not symbol:
            print(f"❌ Invalid symbol provided")
            return None
        
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or "Symbol" not in data:
                print(f"❌ No overview data for {symbol}")
                return None
            
            result = {
                "symbol": data.get("Symbol", symbol),
                "name": data.get("Name", ""),
                "description": data.get("Description", ""),
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "market_cap": self._parse_number(data.get("MarketCapitalization", 0)),
                "pe_ratio": self._parse_float(data.get("PERatio", 0)),
                "peg_ratio": self._parse_float(data.get("PEGRatio", 0)),
                "book_value": self._parse_float(data.get("BookValue", 0)),
                "dividend_yield": self._parse_float(data.get("DividendYield", 0)),
                "eps": self._parse_float(data.get("EPS", 0)),
                "revenue_per_share": self._parse_float(data.get("RevenuePerShareTTM", 0)),
                "profit_margin": self._parse_float(data.get("ProfitMargin", 0)),
                "52_week_high": self._parse_float(data.get("52WeekHigh", 0)),
                "52_week_low": self._parse_float(data.get("52WeekLow", 0)),
                "50_day_ma": self._parse_float(data.get("50DayMovingAverage", 0)),
                "200_day_ma": self._parse_float(data.get("200DayMovingAverage", 0)),
                "beta": self._parse_float(data.get("Beta", 0)),
                "analyst_target": self._parse_float(data.get("AnalystTargetPrice", 0)),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Fetched overview for {symbol}: {result['name']}")
            return result
            
        except Exception as e:
            print(f"❌ Error fetching overview for {symbol}: {str(e)}")
            return None
    
    def fetch_news_sentiment(self, symbol: str, limit: int = 10) -> Optional[list]:
        """
        Fetch news and sentiment for a stock.
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of news items
            
        Returns:
            List of news items or None if failed
        """
        symbol = clean_symbol(symbol)
        if not symbol:
            print(f"❌ Invalid symbol provided")
            return None
        
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": limit,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "feed" not in data:
                print(f"❌ No news data for {symbol}")
                return None
            
            news_items = []
            for item in data["feed"][:limit]:
                # Find sentiment for this specific ticker
                ticker_sentiment = None
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker") == symbol:
                        ticker_sentiment = ts
                        break
                
                news_items.append({
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "time_published": item.get("time_published", ""),
                    "overall_sentiment": item.get("overall_sentiment_label", ""),
                    "overall_sentiment_score": float(item.get("overall_sentiment_score", 0)),
                    "ticker_sentiment": ticker_sentiment.get("ticker_sentiment_label", "") if ticker_sentiment else "",
                    "ticker_sentiment_score": float(ticker_sentiment.get("ticker_sentiment_score", 0)) if ticker_sentiment else 0,
                    "relevance_score": float(ticker_sentiment.get("relevance_score", 0)) if ticker_sentiment else 0,
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                })
            
            print(f"✅ Fetched {len(news_items)} news items for {symbol}")
            return news_items
            
        except Exception as e:
            print(f"❌ Error fetching news for {symbol}: {str(e)}")
            return None
    
    # ==================== CHUNKING ====================
    
    def create_quote_chunk(self, quote_data: dict) -> str:
        """
        Create a text chunk from quote data for embedding.
        
        Args:
            quote_data: Dictionary with quote information
            
        Returns:
            Formatted text string
        """
        change_direction = "up" if quote_data['change'] > 0 else "down"
        
        chunk = f"""
Stock Quote: {quote_data['symbol']}
Current Price: ${quote_data['price']:.2f}
Previous Close: ${quote_data['previous_close']:.2f}
Change: ${quote_data['change']:.2f} ({quote_data['change_percent']}%)
The stock is trading {change_direction} from yesterday.
Day Range: ${quote_data['low']:.2f} - ${quote_data['high']:.2f}
Open: ${quote_data['open']:.2f}
Volume: {quote_data['volume']:,}
Latest Trading Day: {quote_data['latest_trading_day']}
        """.strip()
        
        return chunk
    
    def create_overview_chunks(self, overview_data: dict) -> list[str]:
        """
        Create multiple text chunks from company overview for embedding.
        We split into multiple chunks for better retrieval granularity.
        
        Args:
            overview_data: Dictionary with company information
            
        Returns:
            List of formatted text strings
        """
        chunks = []
        
        # Chunk 1: Basic company info
        basic_info = f"""
Company: {overview_data['name']} ({overview_data['symbol']})
Sector: {overview_data['sector']}
Industry: {overview_data['industry']}
Description: {overview_data['description'][:500]}
        """.strip()
        chunks.append(basic_info)
        
        # Chunk 2: Valuation metrics
        valuation = f"""
{overview_data['name']} ({overview_data['symbol']}) Valuation Metrics:
Market Cap: ${overview_data['market_cap']:,.0f}
P/E Ratio: {overview_data['pe_ratio']:.2f}
PEG Ratio: {overview_data['peg_ratio']:.2f}
Book Value: ${overview_data['book_value']:.2f}
EPS (Earnings Per Share): ${overview_data['eps']:.2f}
Analyst Target Price: ${overview_data['analyst_target']:.2f}
        """.strip()
        chunks.append(valuation)
        
        # Chunk 3: Technical indicators
        technical = f"""
{overview_data['name']} ({overview_data['symbol']}) Technical Indicators:
52-Week High: ${overview_data['52_week_high']:.2f}
52-Week Low: ${overview_data['52_week_low']:.2f}
50-Day Moving Average: ${overview_data['50_day_ma']:.2f}
200-Day Moving Average: ${overview_data['200_day_ma']:.2f}
Beta (Volatility): {overview_data['beta']:.2f}
        """.strip()
        chunks.append(technical)
        
        # Chunk 4: Financial metrics
        financial = f"""
{overview_data['name']} ({overview_data['symbol']}) Financial Metrics:
Dividend Yield: {overview_data['dividend_yield']*100:.2f}%
Profit Margin: {overview_data['profit_margin']*100:.2f}%
Revenue Per Share: ${overview_data['revenue_per_share']:.2f}
        """.strip()
        chunks.append(financial)
        
        return chunks
    
    def create_news_chunk(self, news_item: dict) -> str:
        """
        Create a text chunk from a news item for embedding.
        
        Args:
            news_item: Dictionary with news information
            
        Returns:
            Formatted text string
        """
        chunk = f"""
Stock News for {news_item['symbol']}:
Title: {news_item['title']}
Summary: {news_item['summary'][:400]}
Source: {news_item['source']}
Published: {news_item['time_published']}
Overall Sentiment: {news_item['overall_sentiment']} (score: {news_item['overall_sentiment_score']:.2f})
Ticker Sentiment: {news_item['ticker_sentiment']} (score: {news_item['ticker_sentiment_score']:.2f})
Relevance: {news_item['relevance_score']:.2f}
        """.strip()
        
        return chunk
    
    # ==================== EMBEDDING & STORAGE ====================
    
    def create_embedding(self, text: str) -> list[float]:
        """
        Convert text to embedding vector.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def store_quote(self, quote_data: dict) -> bool:
        """
        Create embedding and store quote in vector database.
        
        Args:
            quote_data: Quote data dictionary
            
        Returns:
            True if successful
        """
        if not quote_data:
            return False
        
        try:
            # Create chunk
            chunk = self.create_quote_chunk(quote_data)
            
            # Create embedding
            embedding = self.create_embedding(chunk)
            
            # Create unique ID with timestamp
            doc_id = f"{quote_data['symbol']}_quote_{quote_data['timestamp']}"
            
            # Store in ChromaDB
            self.quotes_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "symbol": quote_data['symbol'],
                    "price": quote_data['price'],
                    "change": quote_data['change'],
                    "change_percent": float(quote_data['change_percent']),
                    "volume": quote_data['volume'],
                    "timestamp": quote_data['timestamp'],
                    "type": "quote",
                    "full_data": json.dumps(quote_data)
                }]
            )
            
            print(f"   📦 Stored quote embedding for {quote_data['symbol']}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing quote: {str(e)}")
            return False
    
    def store_overview(self, overview_data: dict) -> bool:
        """
        Create embeddings and store company overview chunks in vector database.
        
        Args:
            overview_data: Company overview dictionary
            
        Returns:
            True if successful
        """
        if not overview_data:
            return False
        
        try:
            # Create chunks
            chunks = self.create_overview_chunks(overview_data)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Create embedding
                embedding = self.create_embedding(chunk)
                
                # Create unique ID
                chunk_types = ["basic", "valuation", "technical", "financial"]
                doc_id = f"{overview_data['symbol']}_overview_{chunk_types[i]}"
                
                # Store in ChromaDB
                self.overview_collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "symbol": overview_data['symbol'],
                        "name": overview_data['name'],
                        "sector": overview_data['sector'],
                        "industry": overview_data['industry'],
                        "chunk_type": chunk_types[i],
                        "timestamp": overview_data['timestamp'],
                        "type": "overview",
                        "full_data": json.dumps(overview_data)
                    }]
                )
            
            print(f"   📦 Stored {len(chunks)} overview chunks for {overview_data['symbol']}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing overview: {str(e)}")
            return False
    
    def store_news(self, news_items: list) -> bool:
        """
        Create embeddings and store news items in vector database.
        
        Args:
            news_items: List of news item dictionaries
            
        Returns:
            True if successful
        """
        if not news_items:
            return False
        
        try:
            for i, item in enumerate(news_items):
                # Create chunk
                chunk = self.create_news_chunk(item)
                
                # Create embedding
                embedding = self.create_embedding(chunk)
                
                # Create unique ID
                doc_id = f"{item['symbol']}_news_{item['time_published']}_{i}"
                
                # Store in ChromaDB
                self.news_collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "symbol": item['symbol'],
                        "title": item['title'][:200],
                        "source": item['source'],
                        "sentiment": item['overall_sentiment'],
                        "sentiment_score": item['overall_sentiment_score'],
                        "time_published": item['time_published'],
                        "timestamp": item['timestamp'],
                        "type": "news",
                        "url": item['url']
                    }]
                )
            
            print(f"   📦 Stored {len(news_items)} news embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Error storing news: {str(e)}")
            return False
    
    # ==================== COMPLETE PIPELINE ====================
    
    def process_stock(self, symbol: str, include_news: bool = True) -> dict:
        """
        Complete pipeline: Fetch all data → Create chunks → Embed → Store
        
        Args:
            symbol: Stock ticker symbol
            include_news: Whether to fetch and store news
            
        Returns:
            Dictionary with all fetched data
        """
        symbol = clean_symbol(symbol)
        if not symbol:
            print(f"❌ Invalid symbol provided")
            return {"symbol": symbol, "success": False, "error": "Invalid symbol"}
        
        print(f"\n{'='*60}")
        print(f"📊 Processing {symbol}")
        print('='*60)
        
        result = {"symbol": symbol, "success": False}
        
        # Step 1: Fetch and store quote
        print("\n1️⃣ Fetching real-time quote...")
        quote = self.fetch_quote(symbol)
        if quote:
            self.store_quote(quote)
            result["quote"] = quote
        
        # Step 2: Fetch and store company overview
        print("\n2️⃣ Fetching company overview...")
        overview = self.fetch_company_overview(symbol)
        if overview:
            self.store_overview(overview)
            result["overview"] = overview
        
        # Step 3: Fetch and store news (optional)
        if include_news:
            print("\n3️⃣ Fetching news & sentiment...")
            news = self.fetch_news_sentiment(symbol)
            if news:
                self.store_news(news)
                result["news"] = news
        
        result["success"] = quote is not None or overview is not None
        
        print(f"\n✅ Completed processing {symbol}")
        return result
    
    def process_multiple_stocks(self, symbols: list[str], include_news: bool = False) -> list[dict]:
        """
        Process multiple stocks.
        
        Note: Alpha Vantage free tier has 25 requests/day limit.
        Each stock uses 2-3 API calls (quote + overview + optional news).
        
        Args:
            symbols: List of stock ticker symbols
            include_news: Whether to fetch news (uses extra API calls)
            
        Returns:
            List of results
        """
        results = []
        
        print(f"\n🚀 Processing {len(symbols)} stocks...")
        print("⚠️  Note: Alpha Vantage free tier = 25 requests/day")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}]")
            result = self.process_stock(symbol, include_news=include_news)
            results.append(result)
        
        successful = sum(1 for r in results if r.get("success"))
        print(f"\n{'='*60}")
        print(f"✅ Completed! {successful}/{len(symbols)} stocks processed successfully")
        print('='*60)
        
        return results
    
    # ==================== SEARCH & RETRIEVAL ====================
    
    def search(self, query: str, n_results: int = 5, collection: str = "all") -> list[dict]:
        """
        Search across stock data using natural language.
        
        Args:
            query: Natural language search query
            n_results: Number of results to return
            collection: "quotes", "overviews", "news", or "all"
            
        Returns:
            List of matching results
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        
        results = []
        
        collections_to_search = []
        if collection in ["all", "quotes"]:
            collections_to_search.append(("quotes", self.quotes_collection))
        if collection in ["all", "overviews"]:
            collections_to_search.append(("overviews", self.overview_collection))
        if collection in ["all", "news"]:
            collections_to_search.append(("news", self.news_collection))
        
        for coll_name, coll in collections_to_search:
            try:
                search_results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                
                for i in range(len(search_results['ids'][0])):
                    results.append({
                        "collection": coll_name,
                        "id": search_results['ids'][0][i],
                        "document": search_results['documents'][0][i],
                        "metadata": search_results['metadatas'][0][i],
                        "distance": search_results['distances'][0][i],
                        "similarity": 1 - search_results['distances'][0][i]
                    })
            except Exception:
                continue
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:n_results]
    
    def get_stock_context(self, symbol: str) -> str:
        """
        Get all stored context for a stock (useful for RAG).
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Combined context string
        """
        symbol = clean_symbol(symbol)
        if not symbol:
            return ""
        
        context_parts = []
        
        # Get from quotes collection
        try:
            quotes = self.quotes_collection.get(
                where={"symbol": symbol},
                include=["documents"]
            )
            if quotes['documents']:
                context_parts.extend(quotes['documents'])
        except Exception:
            pass
        
        # Get from overview collection
        try:
            overviews = self.overview_collection.get(
                where={"symbol": symbol},
                include=["documents"]
            )
            if overviews['documents']:
                context_parts.extend(overviews['documents'])
        except Exception:
            pass
        
        # Get from news collection
        try:
            news = self.news_collection.get(
                where={"symbol": symbol},
                include=["documents"]
            )
            if news['documents']:
                context_parts.extend(news['documents'][:5])  # Limit news
        except Exception:
            pass
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        return {
            "quotes_count": self.quotes_collection.count(),
            "overviews_count": self.overview_collection.count(),
            "news_count": self.news_collection.count(),
            "total": (
                self.quotes_collection.count() + 
                self.overview_collection.count() + 
                self.news_collection.count()
            )
        }
    
    # ==================== HELPER METHODS ====================
    
    def _parse_float(self, value) -> float:
        """Safely parse a value to float."""
        try:
            return float(value) if value and value != "None" else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_number(self, value) -> int:
        """Safely parse a value to int."""
        try:
            return int(float(value)) if value and value != "None" else 0
        except (ValueError, TypeError):
            return 0


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🚀 ALPHA VANTAGE → VECTOR DATABASE PIPELINE")
    print("="*70)
    
    # Initialize the pipeline
    db = AlphaVantageVectorDB(db_path="./alpha_vantage_vector_db")
    
    # ===== EXAMPLE 1: Process Single Stock =====
    print("\n📌 EXAMPLE 1: Process Single Stock (AAPL)")
    print("-"*50)
    result = db.process_stock("AAPL", include_news=True)
    
    if result.get("quote"):
        print(f"\n   Quote: ${result['quote']['price']:.2f}")
    if result.get("overview"):
        print(f"   Company: {result['overview']['name']}")
        print(f"   Sector: {result['overview']['sector']}")
    
    # ===== EXAMPLE 2: Process Multiple Stocks =====
    print("\n📌 EXAMPLE 2: Process Multiple Stocks")
    print("-"*50)
    # Note: Be mindful of API rate limits
    portfolio = ["MSFT", "GOOGL"]
    db.process_multiple_stocks(portfolio, include_news=False)
    
    # ===== EXAMPLE 3: Semantic Search =====
    print("\n📌 EXAMPLE 3: Semantic Search")
    print("-"*50)
    
    queries = [
        "technology companies with high PE ratio",
        "stocks that went up today",
        "news about market sentiment"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = db.search(query, n_results=2)
        
        for r in results:
            print(f"   • [{r['collection']}] {r['metadata'].get('symbol', 'N/A')} "
                  f"(similarity: {r['similarity']:.2f})")
    
    # ===== EXAMPLE 4: Get Stock Context (for RAG) =====
    print("\n📌 EXAMPLE 4: Get Stock Context for RAG")
    print("-"*50)
    context = db.get_stock_context("AAPL")
    print(f"   Retrieved {len(context)} characters of context for AAPL")
    
    # ===== EXAMPLE 5: Database Statistics =====
    print("\n📌 EXAMPLE 5: Database Statistics")
    print("-"*50)
    stats = db.get_stats()
    print(f"   Quote embeddings: {stats['quotes_count']}")
    print(f"   Overview embeddings: {stats['overviews_count']}")
    print(f"   News embeddings: {stats['news_count']}")
    print(f"   Total embeddings: {stats['total']}")
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("="*70 + "\n")
