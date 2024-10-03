import pandas as pd
import numpy as np
import scipy.optimize as sco

def compute_covariance_matrix(returns_df): # should be actual returns up until prediction date
    returns_pivot = returns_df.pivot(index='date', columns='permno', values='stock_exret')
    returns_pivot.fillna(0, inplace=True)  # Filling NaNs with 0

    cov_matrix = returns_pivot.cov()
     # Check if the covariance matrix is valid (e.g., no NaNs, infinities)
    if cov_matrix.isna().sum().sum() > 0:
        print("Warning: Covariance matrix contains NaN values.")
    if (cov_matrix == 0).sum().sum() > 0:
        print("Warning: Covariance matrix contains zeros.")
    return cov_matrix

def portfolio_performance(weights, predicted_returns, cov_matrix):
    port_return = np.dot(weights, predicted_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def maximize_sharpe(weights, returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.dot(weights, returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return -sharpe_ratio # optimize function minimizes so here's the negative sharpe lol

def stock_filter(predicted_returns) -> list:
    top_decile_stocks = predicted_returns[predicted_returns >= predicted_returns.quantile(0.9)].index
    bottom_decile_stocks = predicted_returns[predicted_returns <= predicted_returns.quantile(0.1)].index
    filtered_stocks = top_decile_stocks.union(bottom_decile_stocks)

    # Ensure at least 50 stocks are selected
    if len(filtered_stocks) < 50:
        filtered_stocks = predicted_returns.nlargest(50).index
    return filtered_stocks

def optimize_portfolio(predicted_returns_df, actual_returns_df):
    if not np.issubdtype(predicted_returns_df['date'].dtype, np.datetime64):
        predicted_returns_df['date'] = pd.to_datetime(predicted_returns_df['date'], format='%Y%m')
        predicted_returns_df.sort_values('date', inplace=True)

    predicted_pivot = predicted_returns_df.pivot(index='date', columns='permno', values='predicted')
    optimal_weights_list = []
    actual_returns_df = actual_returns_df.loc[:, ['date','permno', 'stock_exret']]
    actual_returns_df['date'] = pd.to_datetime(actual_returns_df['date'], format='%Y%m%d', errors='coerce')
    actual_returns_df.sort_values('date', inplace=True)
    
    for date in predicted_pivot.index:
        shortlisted_stocks = stock_filter(predicted_pivot.loc[date].dropna()) 
        
        # load in historical data and compute covariance matrix
        actual_returns_filterd = actual_returns_df[actual_returns_df['permno'].isin(shortlisted_stocks)]

        actual_returns_till_date = actual_returns_filterd[actual_returns_filterd['date'] < date]

        cov_matrix = compute_covariance_matrix(actual_returns_till_date)

        predicted_returns = predicted_pivot[shortlisted_stocks].loc[date].dropna()
        available_stocks = predicted_returns.index

        # set up and run optimization
        num_assets = len(available_stocks)
        initial_weights = np.array([1 / num_assets] * num_assets)
        bounds = tuple((-1, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        args = (predicted_returns.values, cov_matrix.values)

        optimal_portfolio = sco.minimize(
            maximize_sharpe,
            initial_weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not optimal_portfolio.success:
            print(f"Optimization failed for date {date.strftime('%Y%m')}: {optimal_portfolio.message}")
            continue

        optimal_weights = optimal_portfolio.x
        optimal_weights_series = pd.Series(optimal_weights, index=available_stocks)

        # Flatten the data into a list of dictionaries
        for permno, weight in optimal_weights_series.items():
            optimal_weights_list.append({
                'date': date.strftime('%Y-%m-%d'),  # Convert the date to a string
                'permno': permno,
                'weight': weight
            })

        port_return, port_risk = portfolio_performance(optimal_weights, predicted_returns.values, cov_matrix.values)

    optimal_weights_df = pd.DataFrame(optimal_weights_list)
    
    return optimal_weights_df

if __name__ == "__main__":
    actual_returns_df = pd.read_csv("hackathon_sample_v2.csv")

    predicted_returns_df = pd.read_csv("final_output_20241001_202812.csv") # replace with actual annual predicted data

    # run yearly optimization for all months of that year
    optimal_weights = optimize_portfolio(predicted_returns_df, actual_returns_df)
