import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf 
from scipy.stats import norm 

class VaR:
    def __init__(self, tickers, weights, start_date, end_date, portfolio_value):
        self.tickers = tickers
        self.weights = np.array(weights)
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio_value = portfolio_value

        self.data = None
        self.returns = None
        self.portfolio_mean = None 
        self.portfolio_stdev = None

    def fetch_data(self):
        """ 
        Fetch closing price for the given tickers from Yahoo Finance
        """

        if self.tickers is None:
            raise ValueError("Please send the name of the stocks to be fetched")

        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        self.data = self.data.dropna()

    def calculate_returns(self):
        """
        Compute the returns of the portfolio
        """
        if self.data is None:
            raise ValueError("Please fetch the data first using fetch_data()")
        
        self.returns = (self.data.pct_change()).dropna()
        """
        You can try calculating the log returns as well, but we always consider using the returns over the raw price
        because returns often exhibits stationarity whereas the raw prices doesn't
        """

    def calculate_statistics(self):
        """
        Compute the mean returns.
        """

        if self.returns is None:
            raise ValueError("Calculate the returns using calculate_returns()")

        mean_returns = self.returns.mean()

        """
        If we are considering only one stock, then the standard deviation of its returns is a valid measure 
        of its volatility.
        However, when dealing with a portfolio of multiple stocks, using only the individual standard deviations of 
        the stocks is not sufficient to estimate the portfolio's total risk, because it ignores the correlations 
        between the assets.
        To accurately capture both the individual volatilities and how the stocks move relative to each other, 
        we use the covariance matrix. The portfolio standard deviation (volatility) is then calculated using 
        the weights of the assets and the covariance matrix.
        Portfolio Stdev = √([W]' * [Cov] * [W])
        """
        covariance_matirx = self.returns.cov()

        """
        Now we will compute the portfolio mean and the standard deviation.
        """

        self.portfolio_mean = np.dot(self.weights, mean_returns)
        self.portfolio_stdev = np.sqrt(np.dot(self.weights.T, np.dot(covariance_matirx,self.weights)))

    def calculate_var(self, confidence_level,time_interval=1):
        """
        Compute parametric VaR 
        If we are calculating z-score at 95% CI, then
        VaR = (Portfolio Mean - (Volatility/Portfolio SD * z-score(alpha))) * portfolio Value
        If we are calculating z-score at (1-95%) = 5% CI, then
        VaR = (Portfolio Mean + (Volatility/Portfolio SD * z-score(alpha))) * portfolio Value
        """
        z_score = norm.ppf(1 - confidence_level)
        VaR_1D = (self.portfolio_mean + (self.portfolio_stdev * z_score)) * self.portfolio_value
        return VaR_1D * np.sqrt(time_interval) # Adjust for time interval
    
    def plot_distribution(self, num_simulations=100000, confidence_level=0.95):
        """
        Plot simulated return distribution with VaR highlighted.
        """
        z_score = norm.ppf(1 - confidence_level)
        simulated_returns = np.random.normal(self.portfolio_mean, self.portfolio_stdev, num_simulations)
        simulated_pnl = simulated_returns * self.portfolio_value
        var_value = - (self.portfolio_mean + (self.portfolio_stdev * z_score)) * self.portfolio_value

        plt.figure(figsize=(10,6))
        plt.hist(simulated_pnl, bins=100, alpha=0.6, color='skyblue')
        plt.axvline(- var_value, color='red', linestyle='--', label=f'VaR ({confidence_level*100:.0f}%)')
        plt.title('Simulated Portfolio P&L Distribution')
        plt.xlabel('Daily P&L (₹)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
weights = [0.3,0.3,0.4]
start_date = '2024-08-01'
end_date = '2025-08-01'
portfolio_value = 1000000

"""
Create and run the model
"""
model = VaR(tickers, weights, start_date, end_date, portfolio_value)
model.fetch_data()
model.calculate_returns()
model.calculate_statistics()

var_1d = model.calculate_var(confidence_level=0.99) 
print(f"1-Day VaR at 95% confidence: ₹{var_1d:,.2f}")

# Optional: Plot distribution
model.plot_distribution(confidence_level=0.99)
    


        