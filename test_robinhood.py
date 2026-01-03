"""
Test script for Robinhood Portfolio Analyzer
Run: python test_robinhood.py
"""

from robinhood_portfolio_analyzer import RobinhoodPortfolioAnalyzer

print("="*60)
print("🤖 ROBINHOOD PORTFOLIO ANALYZER TEST")
print("="*60)

# Initialize the analyzer
analyzer = RobinhoodPortfolioAnalyzer()

# Get your credentials
print("\nEnter your Robinhood credentials:")
email = input("Email: ").strip()
password = input("Password: ").strip()
mfa = input("MFA code (press Enter to skip): ").strip() or None

# Login
print("\n🔐 Logging in...")
if analyzer.login(email, password, mfa):
    print("✅ Login successful!")
    
    # Fetch portfolio
    print("\n📊 Fetching your portfolio...")
    portfolio = analyzer.fetch_portfolio()
    
    if portfolio:
        print(f"Found {len(portfolio)} stocks in your portfolio")
        
        # Analyze portfolio
        print("\n🤖 Analyzing with AI...")
        results = analyzer.analyze_portfolio()
        
        # Print the report
        report = analyzer.get_analysis_report(results)
        print(report)
    else:
        print("❌ Could not fetch portfolio")
    
    # Logout
    analyzer.logout()
else:
    print("❌ Login failed. Check your credentials.")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
