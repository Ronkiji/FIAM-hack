import pandas as pd
import numpy as np

def simulate(optimal_weights, actual_returns):

    optimal_weights = optimal_weights.loc[:, ['date', 'permno', 'weight']]
    optimal_weights['date'] = pd.to_datetime(optimal_weights['date'])
    optimal_weights.sort_values('date', inplace=True)

    # actual_returns = actual_returns.loc[:, ['date','permno', 'stock_exret']]
    actual_returns['date'] = pd.to_datetime(actual_returns[['year', 'month']].assign(day=1))
    actual_returns.sort_values('date', inplace=True)

    portfolio_value = 100
    cumulative_values = []

    unique_year_months = sorted(optimal_weights['date'].unique())

    for year_month in unique_year_months:
        
        # Filter portfolio weights and actual returns for the current year-month
        weights = optimal_weights[optimal_weights['date'] == year_month]
        returns = actual_returns[actual_returns['date'] == year_month]

        # Merge weights and actual returns for the same permnos (stocks) based on year-month
        merged_data = pd.merge(weights, returns, on=['permno', 'date'])
        # Calculate the portfolio return for this month
        monthly_return = np.sum(merged_data['weight'] * merged_data['stock_exret'])
        
        # Update portfolio value
        portfolio_value = portfolio_value * (1 + monthly_return)

        # Store the cumulative portfolio value for this year-month
        cumulative_values.append({
            'date': year_month,
            'portfolio_value': portfolio_value,
            'monthly_return': monthly_return
        })

    cumulative_df = pd.DataFrame(cumulative_values)
    cumulative_df.to_csv('cumulative_portfolio_values.csv', index=False)
    return cumulative_df