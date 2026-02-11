# models/volatility.py
import numpy as np
import pandas as pd

def get_ewma_volatility(price_history, decay=0.94, window=252):
    """
    Calculates Exponentially Weighted Moving Average (EWMA) Volatility.
    This gives more weight to recent events (like yesterday's crash)
    and less weight to what happened 10 months ago.
    """
    # Calculate daily log returns
    returns = np.log(price_history / price_history.shift(1)).dropna()
    
    # Initialize variance array
    variance = np.zeros(len(returns))
    
    # Start with simple variance as the initial guess
    variance[0] = returns[:30].var()
    
    # Recursive EWMA calculation
    # Var_t = (lambda * Var_t-1) + ((1 - lambda) * Return_t^2)
    for t in range(1, len(returns)):
        variance[t] = (decay * variance[t-1]) + ((1 - decay) * returns.iloc[t]**2)
        
    # Annualize the latest variance estimate
    # We take the square root to get Volatility (Standard Deviation)
    daily_vol = np.sqrt(variance[-1])
    return daily_vol * np.sqrt(252)