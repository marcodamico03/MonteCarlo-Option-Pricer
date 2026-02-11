# Monte Carlo Option Pricing & Risk Engine

A high-performance **Quantitative Finance Engine** built in Python. This project implements Monte Carlo simulations to price **European** and **Asian (Exotic)** options, calculates **Greeks (Delta, Gamma, Vega)** for risk management, and performs real-time **Volatility Risk Premium (VRP)** analysis on S&P 500 stocks.

Designed to demonstrate the gap between theoretical models (Black-Scholes) and real market behavior (Implied Volatility & Risk Premia).

---

## Methodology

### Monte Carlo Simulation
We model the stock price evolution using **Geometric Brownian Motion (GBM)**:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Where $dW_t$ is a Wiener process. We simulate thousands of paths to approximate the expected payoff at maturity.

### Volatility Models
* **Historical:** Standard deviation of log returns (backward-looking).
* **EWMA (RiskMetrics):** Exponentially Weighted Moving Average. Gives more weight to recent events ($\lambda = 0.94$), making it more responsive to market crashes.
* **Implied:** Reverse-engineered from market option prices using a numerical solver (Brent's method).

---

## Key Features

### 1. Pricing Engine
* **European Options:** Standard Call/Put pricing using Geometric Brownian Motion (GBM).
* **Asian Options (Exotic):** Path-dependent pricing (Arithmetic Average), which Black-Scholes cannot solve analytically.
* **Variance Reduction:** Implemented **Antithetic Variates**, improving convergence speed by **~5.4x**.

### 2. Risk Management (The Greeks)
Calculates sensitivities using the **Finite Difference Method**:
* **Delta ($\Delta$):** Sensitivity to underlying price changes (Hedging).
* **Gamma ($\Gamma$):** Sensitivity of Delta (Convexity).
* **Vega ($\nu$):** Sensitivity to volatility changes.

### 3. Live S&P 500 Analysis
* **Real-Time Scanner:** Connects to Yahoo Finance (`yfinance`) to pull live data for major tickers (AAPL, NVDA, TSLA, etc.).
* **Volatility Arbitrage:** Compares **Historical**, **EWMA (Adaptive)**, and **Implied (Market)** volatilities.
* **Risk Premium (VRP):** Identifies overpriced/underpriced options by calculating the *Volatility Risk Premium*.

### 4. Robust Engineering
* **Offline "Demo Mode":** Automatically detects connection failures and switches to synthetic data generation, ensuring the pipeline never crashes during presentations.

---

## Key Findings & Results

### 1. Variance Reduction Efficiency
Our analysis confirms that **Antithetic Variates** significantly outperforms standard Monte Carlo simulations.
* **Result:** The error rate dropped by approximately **5.4x** for the same number of simulation steps.
* **Implication:** We can achieve accurate pricing with fewer computational resources (or higher accuracy for the same cost).

### 2. Exotic Option Pricing (Asian vs. European)
The engine successfully priced path-dependent Asian Options.
* **Observation:** Asian Call options were consistently priced **lower** than their European counterparts (e.g., ~$15.00 vs ~$29.15 for the same parameters).
* **Reasoning:** The averaging mechanism in Asian options dampens volatility spikes, reducing the probability of extreme payoffs.

### 3. The "Fear Gauge" (Volatility Risk Premium)
By comparing our **EWMA (Adaptive)** model against **Market Implied Volatility**, we identified distinct market regimes:
* **High Fear Regimes:** Stocks like **NVDA** and **TSLA** often showed a positive Risk Premium (>10%), indicating traders are paying a premium for uncertainty (likely due to upcoming earnings).
* **Complacent Regimes:** Stable stocks (e.g., **KO**, **JNJ**) frequently showed near-zero or negative premiums, suggesting options were fairly priced or potentially cheap.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/marcodamico03/MonteCarlo-Option-Pricer]
    cd MonteCarlo-Option-Pricer
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib yfinance scipy
    ```

---

## Usage

Run the main pipeline to execute the full analysis suite:

```bash
python main.py
```

### What happens when you run it?
1.  **Validation:** Benchmarks Monte Carlo results against the Black-Scholes analytical formula.
2.  **Pricing:** Prices a theoretical asset and an Asian Exotic option.
3.  **Risk:** Calculates Delta, Gamma, and Vega.
4.  **Live Scan:** Downloads S&P 500 data (or uses Demo Mode) to generate a **Volatility Battle** report.
5.  **Visualization:** Saves convergence plots, payoff histograms, and risk charts to the `output/` folder.

---

## 📊 Project Structure

```text
├── models/
│   ├── black_scholes.py    # Analytical benchmark formula
│   ├── monte_carlo.py      # Core simulation engine (Pricing + Greeks)
│   └── volatility.py       # EWMA (Exponential Weighted Moving Average) logic
├── output/                 # Generated graphs (Convergence, Histograms, Analysis)
├── main.py                 # Master script that runs the full pipeline
├── sp500_analysis.py       # Live market scanner & VRP calculator
├── settings.py             # Global simulation parameters (Sims, Steps, Seed)
└── README.md               # Project documentation
```

---

### Author
**Marco D'Amico**
*Finance and Asset Management Student*