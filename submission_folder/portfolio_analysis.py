import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *

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
    # Open a file to write output
    with open('output/portfolio_analysis_output.txt', 'w') as file:
        file.write("PORTFOLIO ANALYSIS\n")

        pred = pred.merge(mkt, how="inner", on=["date"])
        nw_ols = sm.ols(formula="monthly_return ~ rf", data=pred).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
        )
        file.write(nw_ols.summary().as_text() + '\n')

        sharpe = pred["monthly_return"].mean() / pred["monthly_return"].std() * np.sqrt(12)
        file.write(f"\nSharpe Ratio (annualized): {sharpe}\n")

        file.write(f"Average Annualized Portfolio Returns: {pred['monthly_return'].mean() * 12}\n")
        file.write(f"Annualized Portfolio Standard Deviation: {pred['monthly_return'].std() * np.sqrt(12)}\n")

        file.write(f"CAPM Alpha: {nw_ols.params['Intercept']}\n")
        file.write(f"Annualized Alpha: {nw_ols.params['Intercept'] * 12}\n")
        file.write(f"t-statistic: {nw_ols.tvalues['Intercept']}\n")
        file.write(f"Information Ratio (annualized): {nw_ols.params['Intercept'] / np.sqrt(nw_ols.mse_resid) * np.sqrt(12)}\n")

        max_1m_loss = pred["monthly_return"].min()
        file.write(f"Max 1-Month Loss: {max_1m_loss}\n")

        pred["log_port"] = np.log(pred["monthly_return"] + 1)
        pred["cumsum_log_port_11"] = pred["log_port"].cumsum(axis=0)
        rolling_peak = pred["cumsum_log_port_11"].cummax()
        drawdowns = rolling_peak - pred["cumsum_log_port_11"]
        max_drawdown = drawdowns.max()
        file.write(f"Maximum Drawdown: {max_drawdown}\n")

        file.write(f"Long & Short Portfolio Turnover: {turnover_count(portfolio_weights)}\n")

        file.write("\nS&P 500 PERFORMANCE\n")
        sharpe_sp = mkt["sp_ret"].mean() / mkt["sp_ret"].std() * np.sqrt(12)
        file.write(f"\nSharpe Ratio (annualized): {sharpe_sp}\n")
        file.write(f"Average Annualized Portfolio Returns: {mkt['sp_ret'].mean() * 12}\n")
        file.write(f"Annualized Portfolio Standard Deviation: {mkt['sp_ret'].std() * np.sqrt(12)}\n")
        file.write(f"Max 1-Month Loss: {mkt['sp_ret'].min()}\n")

        mkt["log_port"] = np.log(mkt["sp_ret"] + 1)
        mkt["cumsum_log_port_11"] = mkt["log_port"].cumsum(axis=0)
        rolling_peak = mkt["cumsum_log_port_11"].cummax()
        drawdowns = rolling_peak - mkt["cumsum_log_port_11"]
        max_drawdown = drawdowns.max()
        file.write(f"Maximum Drawdown: {max_drawdown}\n")