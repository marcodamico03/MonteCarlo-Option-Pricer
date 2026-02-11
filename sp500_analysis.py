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

# Top S&P 500 Stocks
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    "JPM", "BAC", "V", "JNJ", "PFE", "XOM", "CVX", "KO", "PEP", "WMT", "COST"
]

# --- HELPER: BS SOLVER FOR IMPLIED VOLATILITY ---
def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def solve_implied_volatility(market_price, S0, K, T, r):
    def objective(sigma):
        return bs_price(S0, K, T, r, sigma) - market_price
    try:
        return brentq(objective, 0.001, 5.0) 
    except:
        return np.nan

# --- MAIN ANALYSIS LOGIC ---
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Get Data
        hist = stock.history(period="1y")
        if len(hist) < 200: return None
        s0 = hist['Close'].iloc[-1]
        
        # 2. Calculate Volatilities
        # A. Historical
        rets = np.log(hist['Close'] / hist['Close'].shift(1))
        vol_hist = rets.std() * np.sqrt(252)
        
        # B. EWMA
        vol_ewma = get_ewma_volatility(hist['Close'])

        # 3. Get Option Data
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

        chain = stock.option_chain(target_date).calls
        closest = chain.iloc[(np.abs(chain['strike'] - s0)).argmin()]
        K = closest['strike']
        market_price = (closest['bid'] + closest['ask']) / 2
        if market_price == 0: market_price = closest['lastPrice']
        
        # C. Implied Volatility
        T = days / 365.0
        r = 0.045
        vol_implied = solve_implied_volatility(market_price, s0, K, T, r)
        if np.isnan(vol_implied): vol_implied = vol_hist

        # 4. Calculate Prices
        mc1 = MonteCarloEngine(s0, K, T, r, vol_hist, 20000)
        p_hist = mc1.price_call_option(antithetic=True)
        
        mc2 = MonteCarloEngine(s0, K, T, r, vol_ewma, 20000)
        p_ewma = mc2.price_call_option(antithetic=True)

        return {
            "Ticker": ticker,
            # Prices
            "Market_Price": market_price,
            "Price_Hist": p_hist,
            "Price_EWMA": p_ewma,
            "Err_Hist_$": p_hist - market_price,
            "Err_Hist_%": (p_hist - market_price) / market_price * 100,
            "Err_EWMA_$": p_ewma - market_price,
            "Err_EWMA_%": (p_ewma - market_price) / market_price * 100,
            # Volatilities
            "Vol_Implied": vol_implied,
            "Vol_Hist": vol_hist,
            "Vol_EWMA": vol_ewma,
            "Vol_Err_Hist_N": vol_hist - vol_implied, # Diff in points
            "Vol_Err_Hist_P": (vol_hist - vol_implied) / vol_implied * 100,
            "Vol_Err_EWMA_N": vol_ewma - vol_implied,
            "Vol_Err_EWMA_P": (vol_ewma - vol_implied) / vol_implied * 100,
        }
    except Exception as e:
        return None

def run_sector_analysis():
    print(f"--- S&P 500 FINAL ANALYSIS: PRICES & ERRORS ---")
    data = []
    for t in TICKERS:
        print(f"Processing {t}...", end="\r")
        res = analyze_stock(t)
        if res: data.append(res)
            
    df = pd.DataFrame(data)
    pd.options.display.float_format = '{:.2f}'.format
    
    # --- TABLE 1: PRICES & ERRORS ---
    print("\n\n" + "="*100)
    print(" TABLE 1: PRICING ACCURACY ($ and %)")
    print("="*100)
    
    # Rename for cleaner printing
    t1 = df[['Ticker', 'Market_Price', 'Price_Hist', 'Price_EWMA', 
             'Err_Hist_$', 'Err_Hist_%', 'Err_EWMA_$', 'Err_EWMA_%']].copy()
    t1.columns = ['Ticker', 'Mkt $', 'Hist $', 'EWMA $', 
                  'Hist Err $', 'Hist Err %', 'EWMA Err $', 'EWMA Err %']
    
    print(t1.to_string(index=False))

    # --- TABLE 2: VOLATILITY & ERRORS ---
    print("\n\n" + "="*100)
    print(" TABLE 2: VOLATILITY ACCURACY (Number and %)")
    print("="*100)
    
    t2 = df[['Ticker', 'Vol_Implied', 'Vol_Hist', 'Vol_EWMA', 
             'Vol_Err_Hist_N', 'Vol_Err_Hist_P', 'Vol_Err_EWMA_N', 'Vol_Err_EWMA_P']].copy()
    
    # Convert to percentage points for display (0.20 -> 20.0)
    for col in ['Vol_Implied', 'Vol_Hist', 'Vol_EWMA', 'Vol_Err_Hist_N', 'Vol_Err_EWMA_N']:
        t2[col] = t2[col] * 100
        
    t2.columns = ['Ticker', 'Imp Vol', 'Hist Vol', 'EWMA Vol', 
                  'Hist Diff', 'Hist Err %', 'EWMA Diff', 'EWMA Err %']
    
    print(t2.to_string(index=False))

    # --- FINAL SCORECARD (AVERAGES) ---
    avg_hist_err = df['Err_Hist_%'].abs().mean()
    avg_ewma_err = df['Err_EWMA_%'].abs().mean()
    
    print("\n\n" + "="*50)
    print(" FINAL SCORECARD (AVERAGE ABSOLUTE ERROR)")
    print("="*50)
    print(f" 1. Historical Model Avg Error:  {avg_hist_err:.2f}%")
    print(f" 2. EWMA (Smart) Model Avg Error: {avg_ewma_err:.2f}%")
    
    if avg_ewma_err < avg_hist_err:
        print(f" -> WINNER: EWMA Model is {avg_hist_err/avg_ewma_err:.1f}x more accurate.")
    else:
        print(f" -> WINNER: Historical Model (Market is behaving normally).")
    print("="*50)

    # Save Graph
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.35
    plt.bar(x - w/2, df['Err_Hist_%'].abs(), w, label='Hist Error %', color='red', alpha=0.6)
    plt.bar(x + w/2, df['Err_EWMA_%'].abs(), w, label='EWMA Error %', color='blue', alpha=0.6)
    plt.title('Final Accuracy Battle: Historical vs EWMA')
    plt.ylabel('Absolute Price Error (%)')
    plt.xticks(x, df['Ticker'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/5_final_scorecard.png')
    print("\n[+] Final Scorecard Graph saved to output/5_final_scorecard.png")

if __name__ == "__main__":
    run_sector_analysis()