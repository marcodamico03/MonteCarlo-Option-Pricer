# visualize.py
import matplotlib.pyplot as plt
import settings
import numpy as np
from models.monte_carlo import MonteCarloEngine

def plot_paths():
    print("Generating paths... please wait.")
    
    # Use fewer simulations for plotting (otherwise the chart is too messy)
    display_sims = 100 
    
    mc = MonteCarloEngine(
        settings.S0, settings.K, settings.T, settings.R, settings.SIGMA, display_sims
    )
    
    # Generate full paths (252 days)
    paths = mc.simulate_full_paths(steps=252)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(paths) # This draws all 100 lines
    
    # Add the Strike Price line
    plt.axhline(y=settings.K, color='black', linestyle='--', label=f'Strike Price (${settings.K})')
    
    plt.title(f'Monte Carlo Simulation: {display_sims} Random Scenarios')
    plt.xlabel('Trading Days')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Graph generated!")
    plt.show()

if __name__ == "__main__":
    plot_paths()