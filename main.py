# main.py
import os
import settings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime
from models.black_scholes import black_scholes_call
from models.monte_carlo import MonteCarloEngine
from sp500_analysis import run_sector_analysis

def ensure_output_folder():
    if not os.path.exists('output'):
        os.makedirs('output')

# --- VISUALIZATION HELPERS ---
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
    plt.plot(sim_steps, errors_naive, label='Simple Monte Carlo Error', color='#ff9999', linewidth=2)
    plt.plot(sim_steps, errors_smart, label='Antithetic Variates Error', color='#009900', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/1_convergence_plot.png')
    plt.close()

def plot_payoff_histogram():
    print("   -> Generating Payoff Histogram...")
    mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, 50000)
    plt.figure(figsize=(10, 6))
    plt.hist(mc.get_payoff_distribution(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Risk Distribution (Payoffs)')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/2_payoff_histogram.png')
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

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    ensure_output_folder()
    
    print("="*60)
    print(f"OPTION PRICING ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. THEORETICAL BENCHMARK
    bs_price = black_scholes_call(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA)
    print(f"\n[1] BENCHMARK VALIDATION (Black-Scholes): ${bs_price:.4f}")
    
    # 2. SIMULATION ACCURACY
    print(f"[2] ACCURACY CHECK ({settings.N_SIMULATIONS} sims)")
    mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, settings.N_SIMULATIONS)
    
    # Simple MC
    np.random.seed(settings.SEED)
    p_simple = mc.price_call_option(antithetic=False)
    err_simple = abs(p_simple - bs_price)
    pct_simple = (err_simple / bs_price) * 100
    
    # Antithetic MC
    np.random.seed(settings.SEED)
    p_Antithetic = mc.price_call_option(antithetic=True)
    err_Antithetic = abs(p_Antithetic - bs_price)
    pct_Antithetic = (err_Antithetic / bs_price) * 100
    
    print(f"    Simple Monte Carlo:     ${p_simple:.4f} | Error: ${err_simple:.4f} ({pct_simple:.4f}%)")
    print(f"    Antithetic Monte Carlo: ${p_Antithetic:.4f} | Error: ${err_Antithetic:.4f} ({pct_Antithetic:.4f}%)")
    
    if err_Antithetic < err_simple:
        improvement = err_simple / err_Antithetic
        print(f"    -> SUCCESS: Antithetic method reduced error by {improvement:.1f}x")
    
    # 3. EXOTIC OPTION PRICING
    print(f"\n[3] PRICING EXOTIC DERIVATIVES (Asian Option)")
    np.random.seed(settings.SEED)
    p_asian = mc.price_asian_option(antithetic=True)
    print(f"    Asian Call Price:       ${p_asian:.4f}")
    print(f"    European Call Price:    ${p_Antithetic:.4f}")
    print(f"    -> Insight: Asian options are cheaper (averaging reduces volatility).")

    # 4. GENERATING GRAPHS
    print(f"\n[4] GENERATING ANALYTICS GRAPHS...")
    plot_convergence(bs_price)
    plot_payoff_histogram()
    plot_paths()
    print("    [+] Theoretical graphs saved to 'output/'")

    # 5. SECTOR ANALYSIS (Multi Stock)
    print(f"\n[5] STARTING S&P 500 SECTOR SCAN...")
    run_sector_analysis()
    
    print("\n" + "="*60)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY.")