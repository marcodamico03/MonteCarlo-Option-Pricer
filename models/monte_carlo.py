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

    def _generate_paths(self, s_val, sigma_val, antithetic):
        """
        Internal helper to generate price paths.
        Used by both pricing and Greeks calculation to ensure consistency.
        """
        if antithetic:
            # Variance Reduction
            z = np.random.standard_normal(int(self.simulations / 2))
            z = np.concatenate((z, -z))
        else:
            z = np.random.standard_normal(self.simulations)
            
        drift = (self.r - 0.5 * sigma_val ** 2) * self.t
        diffusion = sigma_val * np.sqrt(self.t) * z
        return s_val * np.exp(drift + diffusion)

    def price_call_option(self, antithetic=True):
        """
        Prices the Call Option. 
        Compatible with sp500_analysis.py (accepts antithetic arg).
        """
        st = self._generate_paths(self.s0, self.sigma, antithetic)
        payoff = np.maximum(st - self.k, 0)
        return np.exp(-self.r * self.t) * np.mean(payoff)

    def price_asian_option(self, antithetic=True):
        """Arithmetic Asian Option Pricing"""
        steps = 252
        dt = self.t / steps
        
        if antithetic:
            z = np.random.standard_normal((steps, int(self.simulations / 2)))
            z = np.concatenate((z, -z), axis=1)
        else:
            z = np.random.standard_normal((steps, self.simulations))
            
        st = np.full(self.simulations, self.s0)
        sum_prices = np.zeros(self.simulations)
        
        for t in range(steps):
            z_t = z[t]
            drift = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt) * z_t
            st = st * np.exp(drift + diffusion)
            sum_prices += st
            
        avg_price = sum_prices / steps
        payoff = np.maximum(avg_price - self.k, 0)
        return np.exp(-self.r * self.t) * np.mean(payoff)

    def calculate_greeks(self):
        """
        Calculates Delta, Gamma, Vega using Finite Differences.
        """
        # 1. Delta & Gamma (Bump Price by 1%)
        dS = self.s0 * 0.01
        
        # We assume antithetic=True for stability in Greeks
        state = np.random.get_state() # Save random state
        
        p_base = self.price_call_option(antithetic=True)
        
        np.random.set_state(state) # Reset seed
        p_up = np.mean(np.maximum(self._generate_paths(self.s0 + dS, self.sigma, True) - self.k, 0)) * np.exp(-self.r * self.t)
        
        np.random.set_state(state) # Reset seed
        p_down = np.mean(np.maximum(self._generate_paths(self.s0 - dS, self.sigma, True) - self.k, 0)) * np.exp(-self.r * self.t)
        
        delta = (p_up - p_down) / (2 * dS)
        gamma = (p_up - 2*p_base + p_down) / (dS ** 2)
        
        # 2. Vega (Bump Volatility by 1%)
        dSigma = 0.01
        np.random.set_state(state) # Reset seed
        p_vol_up = np.mean(np.maximum(self._generate_paths(self.s0, self.sigma + dSigma, True) - self.k, 0)) * np.exp(-self.r * self.t)
        
        vega = (p_vol_up - p_base) / (dSigma * 100) # Scaled
        
        return {
            "Price": p_base,
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega
        }

    def get_payoff_distribution(self):
        st = self._generate_paths(self.s0, self.sigma, antithetic=False)
        return np.maximum(st - self.k, 0)

    def simulate_full_paths(self, steps=252):
        dt = self.t / steps
        z = np.random.standard_normal((steps, self.simulations))
        paths = np.zeros((steps + 1, self.simulations))
        paths[0] = self.s0
        for t in range(1, steps + 1):
            drift = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt) * z[t-1]
            paths[t] = paths[t-1] * np.exp(drift + diffusion)
        return paths