# settings.py

# Market Parameters
S0 = 100.0      # Initial Stock Price ($)
K = 100.0       # Strike Price ($)
T = 5.0         # Time to Maturity (1 Year)
R = 0.05        # Risk-Free Rate (5%)
SIGMA = 0.2     # Volatility (20%)

# Simulation Parameters
N_SIMULATIONS = 100_000  # Number of simulated paths
SEED = 42                # Random seed for reproducibility