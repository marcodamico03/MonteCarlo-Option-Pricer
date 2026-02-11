# sp500_analysis.py
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import brentq
from scipy.stats import norm
from models.monte_carlo import MonteCarloEngine
from models.volatility import get_ewma_volatility

# --- CONFIGURATION ---
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", # Tech
    "JPM", "BAC", "V",       # Finance
    "XOM", "CVX",            # Energy
    "JNJ", "PFE",            # Healthcare
    "KO", "PEP", "WMT", "COST" # Consumer
]

# --- HELPER FUNCTIONS ---
def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def solve_implied_volatility(market_price, S0, K, T, r):
    def objective(sigma):
        return bs_price(S0, K, T, r, sigma) - market_price
    try:
        return brentq(objective, 0.01, 5.0) 
    except:
        return np.nan

# --- CORE ANALYSIS ENGINE ---
def analyze_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Get Historical Data (1 Year)
        hist = stock.history(period="1y")
        if len(hist) < 200: return None
        s0 = hist['Close'].iloc[-1]
        
        # 2. Calculate Volatilities
        # A. Historical (Simple Average)
        rets = np.log(hist['Close'] / hist['Close'].shift(1))
        vol_hist = rets.std() * np.sqrt(252)
        
        # B. EWMA (Adaptive/Smart)
        vol_ewma = get_ewma_volatility(hist['Close'])

        # 3. Find Liquid Option (~30 days out)
        expirations = stock.options
        if not expirations: return None
        
        target_date = None
        for date in expirations:
            d = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days
            if 20 < d < 60:
                target_date = date
                days = d
                break
        if not target_date: return None

        # 4. Get Market Price
        chain = stock.option_chain(target_date).calls
        closest = chain.iloc[(np.abs(chain['strike'] - s0)).argmin()]
        K = closest['strike']
        
        market_price = (closest['bid'] + closest['ask']) / 2
        if market_price == 0: market_price = closest['lastPrice']
        
        # 5. Solve Implied Volatility (Market Fear)
        T = days / 365.0
        r = 0.045 
        vol_implied = solve_implied_volatility(market_price, s0, K, T, r)
        if np.isnan(vol_implied): vol_implied = vol_ewma # Fallback

        # 6. Run Monte Carlo Simulations
        # Model 1: Historical
        mc_hist = MonteCarloEngine(s0, K, T, r, vol_hist, 20000)
        p_hist = mc_hist.price_call_option(antithetic=True)
        
        # Model 2: EWMA
        mc_ewma = MonteCarloEngine(s0, K, T, r, vol_ewma, 20000)
        p_ewma = mc_ewma.price_call_option(antithetic=True)
        
        return {
            "Ticker": ticker,
            # Prices
            "Market_Price": market_price,
            "Price_Hist": p_hist,
            "Price_EWMA": p_ewma,
            "Err_Hist_%": (p_hist - market_price) / market_price * 100,
            "Err_EWMA_%": (p_ewma - market_price) / market_price * 100,
            # Volatilities
            "Vol_Implied": vol_implied * 100,
            "Vol_Hist": vol_hist * 100,
            "Vol_EWMA": vol_ewma * 100,
        }

    except Exception as e:
        return None

def run_sector_analysis():
    print(f"--- S&P 500 DYNAMIC ANALYSIS: BATTLE & RISK ---")
    results = []
    for t in TICKERS:
        print(f"   Processing {t}...", end="\r")
        res = analyze_ticker(t)
        if res: results.append(res)
            
    df = pd.DataFrame(results)
    pd.options.display.float_format = '{:.2f}'.format

    # --- PART 1: DECIDE THE WINNER ---
    avg_hist = df['Err_Hist_%'].abs().mean()
    avg_ewma = df['Err_EWMA_%'].abs().mean()
    
    winner = "EWMA" if avg_ewma < avg_hist else "Historical"
    winner_vol_col = 'Vol_EWMA' if winner == "EWMA" else 'Vol_Hist'
    
    # Calculate Risk Premium based on the WINNER
    df['VRP'] = df['Vol_Implied'] - df[winner_vol_col]

    # --- TABLE 1: MODEL BATTLE ---
    print("\n" + "="*95)
    print(" TABLE 1: PRICING ACCURACY BATTLE")
    print("="*95)
    t1 = df[['Ticker', 'Market_Price', 'Price_Hist', 'Price_EWMA', 'Err_Hist_%', 'Err_EWMA_%']].copy()
    t1.columns = ['Ticker', 'Mkt Price $', 'Hist Model $', 'EWMA Model $', 'Hist Err %', 'EWMA Err %']
    print(t1.to_string(index=False))
    
    print("-" * 95)
    print(f" > Avg Error (Historical): {avg_hist:.2f}%")
    print(f" > Avg Error (EWMA Smart): {avg_ewma:.2f}%")
    print(f" 🏆 WINNER: {winner} Model (Used for Risk Premium calculation below)")
    print("="*95)

    # --- TABLE 2: RISK PREMIUM (USING WINNER) ---
    print("\n" + "="*95)
    print(f" TABLE 2: MARKET SENTIMENT (Baseline: {winner} Volatility)")
    print("="*95)
    
    t2 = df[['Ticker', 'Vol_Implied', winner_vol_col, 'VRP']].copy()
    t2.columns = ['Ticker', 'Implied Vol %', f'Fair Vol ({winner}) %', 'Risk Premium (VRP)']
    t2 = t2.sort_values(by='Risk Premium (VRP)', ascending=False)
    
    print(t2.to_string(index=False))
    print("-" * 95)
    print(" > VRP > 0: Market Fear (Implied > Winner).")
    print(" > VRP < 0: Market Complacency (Implied < Winner).")

    # --- GRAPH 1: BATTLE ---
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['Err_Hist_%'].abs(), width, label='Hist Error %', color='red', alpha=0.5)
    plt.bar(x + width/2, df['Err_EWMA_%'].abs(), width, label='EWMA Error %', color='blue', alpha=0.5)
    plt.title('Accuracy Battle: Historical vs EWMA')
    plt.ylabel('Absolute Error (%)')
    plt.xticks(x, df['Ticker'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/4_model_battle.png')
    
    # --- GRAPH 2: RISK PREMIUM ---
    plt.figure(figsize=(12, 6))
    colors = ['orange' if x > 0 else 'green' for x in df['VRP']]
    plt.bar(df['Ticker'], df['VRP'], color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(0, color='black')
    plt.title(f'The Fear Gauge: Implied Volatility vs {winner} Volatility')
    plt.ylabel('Risk Premium (Points)')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/5_risk_premium.png')
    
    print("\n[+] Saved graphs to output/4_model_battle.png and output/5_risk_premium.png")

if __name__ == "__main__":
    run_sector_analysis()