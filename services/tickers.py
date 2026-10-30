"""Company-name and ticker resolution.

Shared by the Streamlit app and the HTTP API, so it must not import streamlit.
"""

# Company name to ticker symbol mapping
COMPANY_TO_TICKER = {
    # Major Tech Companies
    'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 'GOOGLE': 'GOOGL', 'ALPHABET': 'GOOGL',
    'AMAZON': 'AMZN', 'META': 'META', 'FACEBOOK': 'META', 'TESLA': 'TSLA',
    'NVIDIA': 'NVDA', 'NETFLIX': 'NFLX', 'ADOBE': 'ADBE', 'SALESFORCE': 'CRM',
    'ORACLE': 'ORCL', 'INTEL': 'INTC', 'IBM': 'IBM', 'CISCO': 'CSCO',
    'QUALCOMM': 'QCOM', 'AMD': 'AMD', 'PAYPAL': 'PYPL', 'SQUARE': 'SQ',
    'BLOCK': 'SQ', 'TWITTER': 'TWTR', 'SNAP': 'SNAP', 'UBER': 'UBER',
    'LYFT': 'LYFT', 'AIRBNB': 'ABNB', 'SPOTIFY': 'SPOT', 'ZOOM': 'ZM',
    'SHOPIFY': 'SHOP', 'TWILIO': 'TWLO', 'DATADOG': 'DDOG',
    
    # Financial
    'JPMORGAN': 'JPM', 'JP MORGAN': 'JPM', 'BANK OF AMERICA': 'BAC',
    'WELLS FARGO': 'WFC', 'CITIGROUP': 'C', 'GOLDMAN SACHS': 'GS',
    'MORGAN STANLEY': 'MS', 'AMERICAN EXPRESS': 'AXP', 'VISA': 'V',
    'MASTERCARD': 'MA', 'BERKSHIRE HATHAWAY': 'BRK-B', 'BERKSHIRE': 'BRK-B',
    
    # Consumer
    'WALMART': 'WMT', 'TARGET': 'TGT', 'COSTCO': 'COST', 'HOME DEPOT': 'HD',
    'LOWES': 'LOW', 'MCDONALDS': 'MCD', 'STARBUCKS': 'SBUX', 'NIKE': 'NKE',
    'COCA COLA': 'KO', 'PEPSI': 'PEP', 'PEPSICO': 'PEP', 'PROCTER GAMBLE': 'PG',
    'JOHNSON JOHNSON': 'JNJ', 'PFIZER': 'PFE', 'MERCK': 'MRK',
    
    # Auto
    'GENERAL MOTORS': 'GM', 'FORD': 'F', 'TOYOTA': 'TM', 'HONDA': 'HMC',
    
    # Energy
    'EXXON': 'XOM', 'EXXON MOBIL': 'XOM', 'CHEVRON': 'CVX', 'BP': 'BP',
    'SHELL': 'SHEL', 'CONOCOPHILLIPS': 'COP',
    
    # Airlines
    'DELTA': 'DAL', 'AMERICAN AIRLINES': 'AAL', 'UNITED': 'UAL', 'SOUTHWEST': 'LUV',
    
    # Other
    'DISNEY': 'DIS', 'COMCAST': 'CMCSA', 'VERIZON': 'VZ', 'ATT': 'T',
    'AT&T': 'T', 'BOEING': 'BA', 'LOCKHEED MARTIN': 'LMT',
    'CATERPILLAR': 'CAT', '3M': 'MMM', 'HONEYWELL': 'HON'
}

def resolve_ticker(input_str: str) -> tuple[str, bool]:
    """
    Resolve company name or ticker symbol to ticker symbol.
    
    Args:
        input_str: Company name or ticker symbol
        
    Returns:
        Tuple of (ticker_symbol, is_valid)
    """
    if not input_str:
        return '', False
    
    cleaned = input_str.strip().upper()
    # Whitespace-only input cleans to "", which is a substring of every company
    # name -- the partial-match loop below would resolve it to the first entry.
    if not cleaned:
        return '', False

    # Check the company mapping before the ticker-shape heuristic. Short company
    # names (APPLE, TESLA, VISA, FORD...) look like tickers, so testing shape
    # first would pass them through unmapped and fail the quote lookup.
    if cleaned in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[cleaned], True

    # Already a valid ticker format (short, alphanumeric)
    if len(cleaned) <= 5 and cleaned.replace('-', '').isalnum():
        return cleaned, True

    # Try partial matches (e.g., "MICROSOFT CORP" -> "MICROSOFT")
    for company_name, ticker in COMPANY_TO_TICKER.items():
        if company_name in cleaned or cleaned in company_name:
            return ticker, True
    
    # Return as-is if it might be a ticker we don't have mapped
    if len(cleaned) <= 5:
        return cleaned, True
    
    return cleaned, False
