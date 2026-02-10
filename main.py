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
    print("   -> Generating Convergence Plot (this takes a moment)...")
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
    print("   -> Generating Payoff Histogram...")
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
    print("   -> Generating Price Paths Visualization...")
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
    
    # Naive Run
    np.random.seed(settings.SEED)
    p_naive = mc.price_call_option(antithetic=False)
    err_naive = abs(p_naive - bs_price)
    
    # Smart Run
    np.random.seed(settings.SEED)
    p_smart = mc.price_call_option(antithetic=True)
    err_smart = abs(p_smart - bs_price)
    
    print(f"    Naive Monte Carlo Price: ${p_naive:.4f} (Error: {err_naive:.4f})")
    print(f"    Smart Monte Carlo Price: ${p_smart:.4f} (Error: {err_smart:.4f})")
    
    if err_smart < err_naive:
        print(f"    -> SUCCESS: Smart method reduced error by {(err_naive/err_smart):.1f}x")

    # 3. Graphs
    print("\n[3] GENERATING GRAPHS...")
    plot_convergence(bs_price)
    print("    [+] Convergence plot saved.")
    
    plot_payoff_histogram()
    print("    [+] Histogram saved.")

    plot_paths()
    print("    [+] Paths plot saved.")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED. CHECK 'OUTPUT' FOLDER.")