import pandas as pd 
import numpy as np
import yfinance as yf 
import matplotlib.pyplot as plt
from scipy.stats import norm


class VaR:

    def __init__(self,tickers,weights,start_date,end_date,portfolio_value):
        self.tickers = tickers
        self.weights = weights
        self.start_date = start_date
        self.end_date = end_date 
        self.portfolio_value = portfolio_value

        self.data = None 
        self.returns = None
        self.portfolio_returns = None 
        self.var = None

    def fetch_data(self):
        """ 
        Fetch closing price for the given tickers from Yahoo Finance
        """

        if self.tickers is None:
            raise ValueError("Pass valid stock names")

        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        self.data = self.data.dropna()
    
    def calculate_portfolio_returns(self):

        if self.data is None:
            raise ValueError("Fetch stock price data")

        """
        Calculate the stock price returns
        """
        self.returns = (self.data.pct_change()).dropna()

        """
        Calculate the portfolio returns by taking the dot product of the returns and the weights
        """
        self.portfolio_returns = np.dot(self.returns,self.weights)


    def calculate_var(self, confidence_interval):

        if self.portfolio_returns is None:
            raise ValueError("Calculate portfolio returns")

        """ 
        Sort the portfolio returns in ascending order
        """
        self.portfolio_returns = np.sort(self.portfolio_returns)

        """ 
        Calculate aplha = 1-CI 
        (Eg: if CI = 95%, then alpha = 1-0.95 = 0.05 or 5%)
        """
        alpha = 1-confidence_interval
        
        """
        Calculate the number of data points n
        """
        n = len(self.portfolio_returns) - 1

        """ 
        Calculate the rank (Rank = n*(1 - CI) or n*alpha)
        """
        rank = int(n*alpha)

        """ 
        Choose the return at rank position from the sorted returns 
        (Eg: if the rank k = 5, then choose the 5th return(r(k) = r(5)) from the sorted returns)
        """
        return_at_rank = self.portfolio_returns[rank-1]

        """ 
        Calculate VaR
        VaR = r(k) * Portfolio Value
        """
        self.var = return_at_rank*self.portfolio_value

        return self.var
    
    def plot_var(self):
        """Plot portfolio P&L distribution and VaR threshold in currency terms"""
        if self.var is None:
            raise ValueError("Calculate VaR before plotting")

        # Convert returns to P&L in currency
        pnl_values = self.portfolio_returns * self.portfolio_value

        plt.figure(figsize=(10, 6))
        plt.hist(pnl_values, bins=50, alpha=0.6, color='skyblue', edgecolor='black')

        # VaR in currency is already stored in self.var
        plt.axvline(self.var, color='red', linestyle='dotted', linewidth=2,
                    label=f'VaR = {self.var:,.2f}')

        plt.title('Portfolio P&L Distribution with VaR')
        plt.xlabel('Daily P&L')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()



tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
weights = [0.3,0.3,0.4]
start_date = '2024-08-01'
end_date = '2025-08-01'
portfolio_value = 1000000
confidence_interval=0.95

model = VaR(tickers,weights,start_date,end_date,portfolio_value)
model.fetch_data()
model.calculate_portfolio_returns()

VaR = model.calculate_var(confidence_interval)

print(f'1Day VaR at {int(confidence_interval*100)}% CI is, ₹{VaR:,.2f} ')

model.plot_var()