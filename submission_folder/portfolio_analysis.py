import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *


# pred = pd.read_csv('cumulative_portfolio_values.csv', parse_dates=["date"])
# mkt = pd.read_csv('mkt_ind.csv')
# portfolio_weights = pd.read_csv('optimal_weights.csv', parse_dates=["date"])

# Calculate Turnover of the long and short portfolio
def turnover_count(df):
    # count the number of stocks at the begnning of each month
    start_stocks = df[["permno", "date"]].copy()
    start_stocks = start_stocks.sort_values(by=["date", "permno"])
    start_count = start_stocks.groupby(["date"])["permno"].count().reset_index()

    end_stocks = df[["permno", "date"]].copy()
    end_stocks["date"] = end_stocks["date"] - MonthBegin(
        1
    )  # shift the date to the beginning of the next month
    end_stocks = end_stocks.sort_values(by=["date", "permno"])

    remain_stocks = start_stocks.merge(end_stocks, on=["date", "permno"], how="inner")
    remain_count = (
        remain_stocks.groupby(["date"])["permno"].count().reset_index()
    )  # count the number of stocks that remain in the next month
    remain_count = remain_count.rename(columns={"permno": "remain_count"})

    port_count = start_count.merge(remain_count, on=["date"], how="inner")
    port_count["turnover"] = (
        port_count["permno"] - port_count["remain_count"]
    ) / port_count[
        "permno"
    ]  # calculate the turnover as the average of the percentage of stocks that are replaced each month
    return port_count["turnover"].mean()

def analyze_portfolio(pred, mkt, portfolio_weights):
    
    print("PORTFOLIO ANALYSIS") 
    # Calculate the Sharpe ratio for the long-short Portfolio
    # you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately

    # Calculate the CAPM Alpha for the long-short Portfolio
    # you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately

    pred = pred.merge(mkt, how="inner", on=["date"])
    # Newy-West regression for heteroskedasticity and autocorrelation robust standard errors
    nw_ols = sm.ols(formula="monthly_return ~ rf", data=pred).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
    )
    print(nw_ols.summary())

    sharpe = (
        pred["monthly_return"].mean() / pred["monthly_return"].std() * np.sqrt(12)
    )  # Sharpe ratio is annualized
    print("\nSharpe Ratio (annualized):", sharpe)

    print("Average Annualized Portfolio Returns:", pred["monthly_return"].mean() * 12)
    print("Annualized Portfolio Standard Deviation:", pred["monthly_return"].std() * np.sqrt(12))

    # Specifically, the alpha, t-statistic, and Information ratio are:
    print("CAPM Alpha:", nw_ols.params["Intercept"])
    print("Annualized Alpha:", nw_ols.params["Intercept"] * 12)
    print("t-statistic:", nw_ols.tvalues["Intercept"])
    print(
        "Information Ratio (annualized):",
        nw_ols.params["Intercept"] / np.sqrt(nw_ols.mse_resid) * np.sqrt(12),
    )  # Information ratio is annualized

    # Max one-month loss of the long-short Port
    max_1m_loss = pred["monthly_return"].min()
    print("Max 1-Month Loss:", max_1m_loss)

    # Calculate Drawdown of the long-short Portfolio
    pred["log_port"] = np.log(
        pred["monthly_return"] + 1
    )  # calculate log returns
    pred["cumsum_log_port_11"] = pred["log_port"].cumsum(
        axis=0
    )  # calculate cumulative log returns
    rolling_peak = pred["cumsum_log_port_11"].cummax()
    drawdowns = rolling_peak - pred["cumsum_log_port_11"]
    max_drawdown = drawdowns.max()
    print("Maximum Drawdown:", max_drawdown)

    print("Long & Short Portfolio Turnover:", turnover_count(portfolio_weights))

    print("\nS&P 500 PERFORMANCE")
    sharpe_sp = (
        mkt["sp_ret"].mean() / mkt["sp_ret"].std() * np.sqrt(12)
    )  # Sharpe ratio is annualized
    print("\nSharpe Ratio (annualized):", sharpe_sp)
    print("Average Annualized Portfolio Returns:", mkt["sp_ret"].mean() * 12)
    print("Annualized Portfolio Standard Deviation:", mkt["sp_ret"].std() * np.sqrt(12))
    print("Max 1-Month Loss:", mkt["sp_ret"].min())
    # Calculate Drawdown of the long-short Portfolio
    mkt["log_port"] = np.log(
        mkt["sp_ret"] + 1
    )  # calculate log returns
    mkt["cumsum_log_port_11"] = mkt["log_port"].cumsum(
        axis=0
    )  # calculate cumulative log returns
    rolling_peak = mkt["cumsum_log_port_11"].cummax()
    drawdowns = rolling_peak - mkt["cumsum_log_port_11"]
    max_drawdown = drawdowns.max()
    print("Maximum Drawdown:", max_drawdown)
