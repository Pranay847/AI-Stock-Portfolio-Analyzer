import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class MistralStockAnalyzer:
    """
    Stock analyzer using Mistral AI.
    Supports both local (Ollama) and cloud (Mistral API) options.
    """
    
    def __init__(self, use_ollama: bool = True, vector_db_path: str = "./stock_vector_db"):
        """
        Initialize the Mistral analyzer.
        
        Args:
            use_ollama: If True, use local Ollama. If False, use Mistral API.
            vector_db_path: Path to the vector database for RAG context.
        """
        self.use_ollama = use_ollama
        self.model_name = "mistral"  # For Ollama
        
        # Initialize Mistral client based on mode
        if use_ollama:
            self._init_ollama()
        else:
            self._init_mistral_api()
        
        # Initialize vector database for RAG (optional but recommended)
        self.vector_db = None
        try:
            from alpha_vantage_vector_db import AlphaVantageVectorDB
            self.vector_db = AlphaVantageVectorDB(db_path=vector_db_path)
            print("✅ Vector database connected (RAG enabled)")
        except Exception as e:
            print(f"⚠️  Vector database not available: {e}")
            print("   Analysis will work but without historical context")
    
    def _init_ollama(self):
        """Initialize Ollama client for local Mistral."""
        try:
            import ollama
            self.ollama_client = ollama
            # Check if Mistral is available
            models = ollama.list()
            model_names = [m.get('name', '') for m in models.get('models', [])]
            
            if not any('mistral' in name.lower() for name in model_names):
                print("⚠️  Mistral not found in Ollama. Run: ollama pull mistral")
                print("   Available models:", model_names)
            else:
                print("✅ Ollama + Mistral ready")
                
        except ImportError:
            print("❌ Ollama not installed. Run: pip install ollama")
            self.ollama_client = None
        except Exception as e:
            print(f"⚠️  Ollama not running: {e}")
            print("   Start Ollama with: ollama serve")
            self.ollama_client = None
    
    def _init_mistral_api(self):
        """Initialize Mistral API client (cloud)."""
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        if not self.mistral_api_key:
            print("❌ MISTRAL_API_KEY not found in .env")
            print("   Get your key at: https://console.mistral.ai/")
            self.mistral_client = None
            return
        
        try:
            from mistralai import Mistral
            self.mistral_client = Mistral(api_key=self.mistral_api_key)
            self.model_name = "mistral-small-latest"  # or "mistral-large-latest"
            print("✅ Mistral API ready")
        except ImportError:
            print("❌ mistralai not installed. Run: pip install mistralai")
            self.mistral_client = None
    
    # ==================== RAG: GET CONTEXT ====================
    
    def get_stock_context(self, symbol: str) -> str:
        """
        Get relevant context for a stock using RAG.
        This retrieves data from the vector database.
        """
        if not self.vector_db:
            return ""
        
        try:
            # Get stored context from vector DB
            context = self.vector_db.get_stock_context(symbol)
            
            if not context:
                # Try to fetch fresh data
                print(f"   📡 Fetching data for {symbol}...")
                self.vector_db.process_stock(symbol, include_news=False)
                context = self.vector_db.get_stock_context(symbol)
            
            return context
        except Exception as e:
            print(f"   ⚠️  Could not get context: {e}")
            return ""
    
    def search_similar_stocks(self, query: str, n_results: int = 3) -> str:
        """
        Search for similar stocks/context using semantic search.
        """
        if not self.vector_db:
            return ""
        
        try:
            results = self.vector_db.search(query, n_results=n_results)
            if results:
                context_parts = [r['document'] for r in results]
                return "\n\n".join(context_parts)
            return ""
        except Exception:
            return ""
    
    # ==================== MISTRAL AI CALLS ====================
    
    def _call_mistral(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call Mistral AI (either Ollama or API).
        """
        if self.use_ollama:
            return self._call_ollama(prompt, system_prompt)
        else:
            return self._call_mistral_api(prompt, system_prompt)
    
    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Call Mistral via Ollama (local)."""
        if not self.ollama_client:
            return '{"error": "Ollama not available"}'
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=messages
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f'{{"error": "Ollama error: {str(e)}"}}'
    
    def _call_mistral_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call Mistral via cloud API."""
        if not self.mistral_client:
            return '{"error": "Mistral API not available"}'
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.mistral_client.chat.complete(
                model=self.model_name,
                messages=messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f'{{"error": "Mistral API error: {str(e)}"}}'
    
    # ==================== STOCK ANALYSIS ====================
    
    def analyze_stock(self, symbol: str, position: dict = None) -> dict:
        """
        Analyze a stock using Mistral AI with RAG context.
        
        Args:
            symbol: Stock ticker symbol
            position: Optional dict with your position data:
                      {quantity, average_buy_price, current_price, profit_loss_percent}
        
        Returns:
            Analysis result with recommendation
        """
        print(f"\n🔍 Analyzing {symbol} with Mistral AI...")
        
        # Step 1: Get context from vector database (RAG)
        context = self.get_stock_context(symbol)
        
        # Step 2: Build the prompt
        system_prompt = """You are an expert financial analyst. Analyze stocks and provide clear recommendations.
Always respond in valid JSON format with this structure:
{
    "recommendation": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "summary": "2-3 sentence explanation",
    "reasons": ["reason 1", "reason 2", "reason 3"],
    "target_price": null or number,
    "risk_level": "LOW" or "MEDIUM" or "HIGH"
}"""
        
        # Build user prompt with context
        prompt_parts = [f"Analyze the stock: {symbol}"]
        
        if context:
            prompt_parts.append(f"\n\nHere is the current market data and context:\n{context}")
        
        if position:
            prompt_parts.append(f"""

My current position:
- Shares owned: {position.get('quantity', 0)}
- Average buy price: ${position.get('average_buy_price', 0):.2f}
- Current price: ${position.get('current_price', 0):.2f}
- Profit/Loss: {position.get('profit_loss_percent', 0):.2f}%
""")
        
        prompt_parts.append("""
Based on this information, should I BUY more, SELL, or HOLD this position?
Consider technical indicators, valuation, and risk. Respond in JSON format.""")
        
        prompt = "\n".join(prompt_parts)
        
        # Step 3: Call Mistral
        print("   🤖 Asking Mistral AI...")
        response = self._call_mistral(prompt, system_prompt)
        
        # Step 4: Parse response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            result['symbol'] = symbol
            result['analysis_type'] = 'mistral_ai'
            print(f"   ✅ {result.get('recommendation', 'HOLD')} (confidence: {result.get('confidence', 0)}%)")
            return result
            
        except json.JSONDecodeError:
            # If JSON parsing fails, extract info from text
            print(f"   ⚠️  Could not parse JSON, using text response")
            recommendation = 'HOLD'
            if 'BUY' in response.upper():
                recommendation = 'BUY'
            elif 'SELL' in response.upper():
                recommendation = 'SELL'
            
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': 50,
                'summary': response[:500],
                'reasons': [],
                'analysis_type': 'mistral_ai_text'
            }
    
    def analyze_portfolio(self, positions: list) -> list:
        """
        Analyze multiple stocks in a portfolio.
        
        Args:
            positions: List of position dicts
            
        Returns:
            List of analysis results
        """
        print(f"\n{'='*60}")
        print(f"🤖 MISTRAL AI PORTFOLIO ANALYSIS")
        print(f"   Analyzing {len(positions)} stocks...")
        print('='*60)
        
        results = []
        for i, position in enumerate(positions, 1):
            symbol = position.get('symbol', 'UNKNOWN')
            print(f"\n[{i}/{len(positions)}] {symbol}")
            
            result = self.analyze_stock(symbol, position)
            results.append(result)
        
        # Summary
        buy_count = sum(1 for r in results if r.get('recommendation') == 'BUY')
        sell_count = sum(1 for r in results if r.get('recommendation') == 'SELL')
        hold_count = sum(1 for r in results if r.get('recommendation') == 'HOLD')
        
        print(f"\n{'='*60}")
        print("📊 SUMMARY")
        print(f"   🟢 BUY:  {buy_count}")
        print(f"   🔴 SELL: {sell_count}")
        print(f"   🟡 HOLD: {hold_count}")
        print('='*60)
        
        return results
    
    def chat_about_stocks(self, question: str) -> str:
        """
        Have a conversation about stocks with Mistral.
        Uses RAG to provide context.
        
        Args:
            question: Your question about stocks
            
        Returns:
            Mistral's response
        """
        # Search for relevant context
        context = self.search_similar_stocks(question, n_results=5)
        
        system_prompt = """You are a helpful financial assistant. 
Answer questions about stocks clearly and concisely.
If you don't know something, say so."""
        
        prompt = question
        if context:
            prompt = f"""Context from our stock database:
{context}

User question: {question}

Please answer based on the context above and your knowledge."""
        
        return self._call_mistral(prompt, system_prompt)


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 MISTRAL AI STOCK ANALYZER")
    print("="*70)
    
    print("\nChoose mode:")
    print("1. Use Ollama (free, local) - requires: ollama pull mistral")
    print("2. Use Mistral API (cloud) - requires: MISTRAL_API_KEY in .env")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    use_ollama = choice != "2"
    
    # Initialize analyzer
    analyzer = MistralStockAnalyzer(use_ollama=use_ollama)
    
    # Example 1: Analyze a single stock
    print("\n" + "-"*50)
    print("📌 EXAMPLE 1: Single Stock Analysis")
    print("-"*50)
    
    result = analyzer.analyze_stock("AAPL", {
        'quantity': 10,
        'average_buy_price': 150.00,
        'current_price': 175.00,
        'profit_loss_percent': 16.67
    })
    
    print(f"\nResult for AAPL:")
    print(f"   Recommendation: {result.get('recommendation')}")
    print(f"   Confidence: {result.get('confidence')}%")
    print(f"   Summary: {result.get('summary', 'N/A')[:200]}")
    
    # Example 2: Chat about stocks
    print("\n" + "-"*50)
    print("📌 EXAMPLE 2: Chat with Mistral")
    print("-"*50)
    
    response = analyzer.chat_about_stocks("What technology stocks look good right now?")
    print(f"\nMistral says:\n{response[:500]}...")
    
    # Example 3: Portfolio analysis
    print("\n" + "-"*50)
    print("📌 EXAMPLE 3: Portfolio Analysis")
    print("-"*50)
    
    mock_portfolio = [
        {'symbol': 'AAPL', 'quantity': 10, 'average_buy_price': 150, 'current_price': 175, 'profit_loss_percent': 16.67},
        {'symbol': 'TSLA', 'quantity': 5, 'average_buy_price': 200, 'current_price': 250, 'profit_loss_percent': 25.0},
    ]
    
    results = analyzer.analyze_portfolio(mock_portfolio)
    
    print("\n✅ Analysis complete!")
