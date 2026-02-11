# sp500_analysis.py
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from models.monte_carlo import MonteCarloEngine

# Representative list of major S&P 500 sectors
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", # Tech
    "JPM", "BAC", "V",       # Finance
    "JNJ", "PFE",            # Healthcare
    "XOM", "CVX",            # Energy
    "KO", "PEP",             # Consumer
    "WMT", "COST"            # Retail
]

def get_data_and_price(ticker):
    try:
        # 1. Historical Volatility
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if len(hist) < 200: return None
        
        rets = np.log(hist['Close'] / hist['Close'].shift(1))
        sigma = rets.std() * np.sqrt(252)
        s0 = hist['Close'].iloc[-1]
        
        # 2. Find Option (~30 days out)
        expirations = stock.options
        if not expirations: return None
        
        target_date = None
        days_to_maturity = 0
        for date in expirations:
            d = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days
            if 20 < d < 50:
                target_date = date
                days_to_maturity = d
                break
        
        if not target_date: return None

        # 3. ATM Option Price
        chain = stock.option_chain(target_date).calls
        closest_idx = (np.abs(chain['strike'] - s0)).argmin()
        row = chain.iloc[closest_idx]
        
        K = row['strike']
        market_price = (row['bid'] + row['ask']) / 2
        if market_price == 0: market_price = row['lastPrice']
        
        # 4. Pricing
        T = days_to_maturity / 365.0
        r = 0.045 
        mc = MonteCarloEngine(s0, K, T, r, sigma, simulations=20000)
        model_price = mc.price_call_option(antithetic=True)
        
        return {
            "Ticker": ticker,
            "Sigma_Hist": sigma,
            "Market_Price": market_price,
            "Model_Price": model_price,
            "Diff_Pct": (model_price - market_price) / market_price * 100
        }
    except:
        return None

def run_sector_analysis():
    print(f"--- RUNNING S&P 500 SECTOR SCAN ({len(TICKERS)} Stocks) ---")
    results = []
    
    for i, t in enumerate(TICKERS):
        print(f"   Processing [{i+1}/{len(TICKERS)}] {t}...", end="\r")
        res = get_data_and_price(t)
        if res: results.append(res)
            
    print(f"\n   Analysis Complete. Generating Report...")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Absolute Difference ($)
    df['Diff_Abs'] = df['Model_Price'] - df['Market_Price']
    
    # Select and Rename columns for the Table
    table_view = df[['Ticker', 'Model_Price', 'Market_Price', 'Diff_Abs', 'Diff_Pct']].copy()
    table_view.columns = ['Ticker', 'Model ($)', 'Market ($)', 'Diff ($)', 'Diff (%)']
    
    # Print the Table formatted cleanly
    print("\n" + "="*65)
    print("S&P 500 PRICING REPORT (Model vs Market)")
    print("="*65)
    # This prints the table without the index number, rounded to 2 decimals
    print(table_view.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*65 + "\n")

    # --- Generate the Scatter Plot (Same as before) ---
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in df['Diff_Pct']]
    plt.scatter(df['Sigma_Hist'], df['Diff_Pct'], c=colors, s=100, alpha=0.7, edgecolors='black')
    
    for i, txt in enumerate(df['Ticker']):
        plt.annotate(txt, (df['Sigma_Hist'].iloc[i], df['Diff_Pct'].iloc[i]), xytext=(5,5), textcoords='offset points')
        
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Monte Carlo vs Market Prices: The Volatility Premium')
    plt.xlabel('Historical Volatility (1 Year)')
    plt.ylabel('Model Deviation from Market (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/4_sp500_analysis.png')
    plt.close()
    print("   [+] Scatter plot saved to output/4_sp500_analysis.png")