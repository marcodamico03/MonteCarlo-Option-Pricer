# main.py
import os
import settings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from models.black_scholes import black_scholes_call
from models.monte_carlo import MonteCarloEngine
from sp500_analysis import run_sector_analysis 

def ensure_output_folder():
    if not os.path.exists('output'):
        os.makedirs('output')

def plot_convergence(bs_price):
    print("   -> Generating Convergence Plot...")
    sim_steps = range(1000, 51000, 2000)
    errors_naive, errors_smart = [], []

    for n in sim_steps:
        # Naive
        np.random.seed(42)
        mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, n)
        errors_naive.append(abs(mc.price_call_option(antithetic=False) - bs_price))
        # Smart
        np.random.seed(42)
        mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, n)
        errors_smart.append(abs(mc.price_call_option(antithetic=True) - bs_price))

    plt.figure(figsize=(10, 6))
    plt.plot(sim_steps, errors_naive, label='Simple MC Error', color='#ff9999', linewidth=2)
    plt.plot(sim_steps, errors_smart, label='Antithetic Error', color='#009900', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.savefig('output/1_convergence_plot.png')
    plt.close()

def plot_paths():
    print("   -> Generating Price Paths...")
    mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, 100)
    plt.figure(figsize=(10, 6))
    plt.plot(mc.simulate_full_paths(), alpha=0.6)
    plt.axhline(y=settings.K, color='black', linestyle='--')
    plt.title('Monte Carlo Simulation Paths')
    plt.savefig('output/3_simulation_paths.png')
    plt.close()

if __name__ == "__main__":
    ensure_output_folder()
    
    print("="*60)
    print(f"QUANTITATIVE RISK ENGINE v3.1 (Fixed) - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # 1. BENCHMARK
    bs_price = black_scholes_call(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA)
    print(f"\n[1] THEORETICAL BENCHMARK: ${bs_price:.4f}")
    
    # 2. PRICING & GREEKS
    print(f"[2] CALCULATING PRICE & GREEKS...")
    np.random.seed(settings.SEED)
    mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, settings.N_SIMULATIONS)
    
    # Calculate everything
    greeks = mc.calculate_greeks()
    price = greeks['Price'] # We get price for free alongside Greeks
    
    print(f"    Monte Carlo Price: ${price:.4f}")
    print("-" * 40)
    print(f"    DELTA (Stock Sensitivity): {greeks['Delta']:.4f}")
    print(f"    GAMMA (Delta Sensitivity): {greeks['Gamma']:.4f}")
    print(f"    VEGA  (Vol Sensitivity):   {greeks['Vega']:.4f}")
    print("-" * 40)

    # 3. EXOTIC (Asian)
    print(f"\n[3] PRICING ASIAN OPTION...")
    np.random.seed(settings.SEED)
    asian_price = mc.price_asian_option(antithetic=True)
    print(f"    Asian Price: ${asian_price:.4f}")

    # 4. GRAPHS
    print(f"\n[4] GENERATING CHARTS...")
    plot_convergence(bs_price)
    plot_paths()
    print("    [+] Charts saved.")

    # 5. LIVE MARKET (This will now work!)
    print(f"\n[5] LAUNCHING S&P 500 ANALYSIS...")
    run_sector_analysis()
    
    print("\n" + "="*60)
    print("PIPELINE SUCCESSFUL.")