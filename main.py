# main.py
import os
import settings
import numpy as np
import matplotlib.pyplot as plt
from models.black_scholes import black_scholes_call
from models.monte_carlo import MonteCarloEngine

def ensure_output_folder():
    if not os.path.exists('output'):
        os.makedirs('output')

def plot_convergence(bs_price):
    print("Generating Convergence Plot...")
    sim_steps = range(1000, 51000, 2000)
    errors_naive = []
    errors_smart = []

    for n in sim_steps:
        np.random.seed(42)
        mc_naive = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, n)
        price_n = mc_naive.price_call_option(antithetic=False)
        errors_naive.append(abs(price_n - bs_price))
        
        np.random.seed(42)
        mc_smart = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, n)
        price_s = mc_smart.price_call_option(antithetic=True)
        errors_smart.append(abs(price_s - bs_price))

    plt.figure(figsize=(10, 6))
    plt.plot(sim_steps, errors_naive, label='Naive Monte Carlo Error', color='#ff9999', linewidth=2)
    plt.plot(sim_steps, errors_smart, label='Antithetic Variates Error (Smart)', color='#009900', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('Convergence Analysis: Naive vs Antithetic')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Absolute Error ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/1_convergence_plot.png')
    plt.close()

def plot_payoff_histogram():
    print("Generating Payoff Histogram...")
    mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, simulations=50000)
    payoffs = mc.get_payoff_distribution()
    
    plt.figure(figsize=(10, 6))
    plt.hist(payoffs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Option Payoff Distribution (Maturity)')
    plt.xlabel('Payoff Value ($)')
    plt.ylabel('Frequency')
    plt.axvline(x=np.mean(payoffs), color='red', linestyle='--', label=f'Mean Payoff: ${np.mean(payoffs):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/2_payoff_histogram.png')
    plt.close()

def plot_paths():
    print("Generating Price Paths Visualization...")
    mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, simulations=100)
    paths = mc.simulate_full_paths(steps=252)

    plt.figure(figsize=(10, 6))
    plt.plot(paths, alpha=0.6, linewidth=1)
    plt.axhline(y=settings.K, color='black', linestyle='--', label=f'Strike Price (${settings.K})')
    plt.title(f'Monte Carlo Simulation: 100 Random Paths')
    plt.xlabel('Trading Days')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/3_simulation_paths.png')
    plt.close()

if __name__ == "__main__":
    ensure_output_folder()
    
    print("="*60)
    print(f"MONTE CARLO OPTION PRICER (Asset=${settings.S0}, Strike=${settings.K})")
    print("="*60)
    
    # 1. Benchmark
    bs_price = black_scholes_call(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA)
    print(f"[1] BENCHMARK (Black-Scholes): ${bs_price:.4f}")
    
    # 2. Final Price Check (The Analysis you wanted!)
    print(f"\n[2] FINAL SIMULATION ({settings.N_SIMULATIONS} scenarios)")
    mc = MonteCarloEngine(settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, settings.N_SIMULATIONS)

    # Simple Run
    np.random.seed(settings.SEED)
    p_simple = mc.price_call_option(antithetic=False)
    err_simple = abs(p_simple - bs_price)
    
    # Antithetic Run
    np.random.seed(settings.SEED)
    p_Antithetic = mc.price_call_option(antithetic=True)
    err_Antithetic = abs(p_Antithetic - bs_price)
    
    print(f"    Simple Monte Carlo Price: ${p_simple:.4f} (Error: {err_simple:.4f})")
    print(f"    Antithetic Monte Carlo Price: ${p_Antithetic:.4f} (Error: {err_Antithetic:.4f})")
    
    if err_Antithetic < err_simple:
        print(f"SUCCESS: Antithetic method reduced error by {(err_simple/err_Antithetic):.1f}x")

    print("-" * 60)
    print("REAL WORLD APPLICATION: EXOTIC PRICING (ASIAN OPTION)")
    print("-" * 60)
    
    # Prezzo Asiatica
    np.random.seed(settings.SEED)
    p_asian = mc.price_asian_option(antithetic=True)
    
    print(f"    Asian Option Price:           ${p_asian:.4f}")
    print(f"    European Option Price:        ${p_Antithetic:.4f}")
    print(f"    -> Insight: Asian options are cheaper because averaging reduces volatility.")

    # 3. Graphs
    print("\n[3] GENERATING GRAPHS...")
    plot_convergence(bs_price)
    print("Convergence plot saved.")
    
    plot_payoff_histogram()
    print("Histogram saved.")

    plot_paths()
    print("Paths plot saved.")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED. CHECK 'OUTPUT' FOLDER.")