# Portfolio Value at Risk (VaR) Calculation

## Objective
This script calculates the **1-Day Value at Risk (VaR)** for a portfolio of stocks using the 
1. **Historical Method**
2. **Varince-Covariance Method**  
The **VaR** represents the potential loss in portfolio value over a given time horizon (here: 1 day) at a specific confidence interval (e.g., 95%).  
It also visualizes the distribution of simulated daily portfolio P&L values with the VaR threshold marked.

---

### **Import Required Libraries**
- `pandas` and `numpy` for data manipulation and calculations.
- `yfinance` to fetch historical stock price data.
- `matplotlib` for plotting results.
- `scipy.stats` (imported for possible statistical extensions).

---

### **Define the `VaR` Class**
The class encapsulates the entire process:
- **Initialization (`__init__`)**  
  Stores tickers, weights, date range, portfolio value, and placeholders for data, returns, and VaR.

---

