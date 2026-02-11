# models/monte_carlo.py
import numpy as np

class MonteCarloEngine:
    def __init__(self, s0, k, t, r, sigma, simulations):
        self.s0 = s0
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        self.simulations = simulations
    
    def price_call_option(self, antithetic=True):
        """
        Prices a European Call Option (Depends only on final price ST).
        """
        if antithetic:
            # Variance Reduction: Generate half, then mirror them
            z = np.random.standard_normal(int(self.simulations / 2))
            z_combined = np.concatenate((z, -z))
        else:
            z_combined = np.random.standard_normal(self.simulations)
            
        drift = (self.r - 0.5 * self.sigma ** 2) * self.t
        diffusion = self.sigma * np.sqrt(self.t) * z_combined
        st = self.s0 * np.exp(drift + diffusion)
        
        payoff = np.maximum(st - self.k, 0)
        return np.exp(-self.r * self.t) * np.mean(payoff)

    def price_asian_option(self, antithetic=True):
        """
        Prices an Arithmetic Asian Call Option (Depends on Average Price).
        Black-Scholes cannot calculate this directly.
        """
        steps = 252 # Daily monitoring
        dt = self.t / steps
        
        if antithetic:
            z = np.random.standard_normal((steps, int(self.simulations / 2)))
            z_combined = np.concatenate((z, -z), axis=1)
        else:
            z_combined = np.random.standard_normal((steps, self.simulations))
            
        # Simulate full paths
        st = np.full(self.simulations, self.s0)
        sum_prices = np.zeros(self.simulations)
        
        for t in range(steps):
            z_t = z_combined[t]
            drift = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt) * z_t
            st = st * np.exp(drift + diffusion)
            sum_prices += st
            
        # Payoff depends on the AVERAGE price, not the final price
        average_price = sum_prices / steps
        payoff = np.maximum(average_price - self.k, 0)
        
        return np.exp(-self.r * self.t) * np.mean(payoff)

    def get_payoff_distribution(self):
        """Returns raw payoffs for histogram plotting."""
        z = np.random.standard_normal(self.simulations)
        drift = (self.r - 0.5 * self.sigma ** 2) * self.t
        diffusion = self.sigma * np.sqrt(self.t) * z
        st = self.s0 * np.exp(drift + diffusion)
        return np.maximum(st - self.k, 0)

    def simulate_full_paths(self, steps=252):
        """Returns full paths for visualization."""
        dt = self.t / steps
        z = np.random.standard_normal((steps, self.simulations))
        paths = np.zeros((steps + 1, self.simulations))
        paths[0] = self.s0
        
        for t in range(1, steps + 1):
            drift = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt) * z[t-1]
            paths[t] = paths[t-1] * np.exp(drift + diffusion)
            
        return paths