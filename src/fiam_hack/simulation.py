import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Change as needed
start_date = '2012-01-01'
end_date = '2012-11-01'

optimal_weights = pd.read_csv('optimal_weights.csv')
optimal_weights = optimal_weights.loc[:, ['date', 'permno', 'weight']]
optimal_weights['date'] = pd.to_datetime(optimal_weights['date'])

actual_returns = pd.read_csv('hackathon_sample_v2.csv')
actual_returns = actual_returns.loc[:, ['year','month','permno', 'stock_exret']]
actual_returns['date'] = pd.to_datetime(actual_returns[['year', 'month']].assign(day=1))

portfolio_value = 100
cumulative_values = []

unique_year_months = sorted(optimal_weights['date'].unique())

for year_month in unique_year_months:
    
    # Filter portfolio weights and actual returns for the current year-month
    weights = optimal_weights[optimal_weights['date'] == year_month]
    returns = actual_returns[actual_returns['date'] == year_month]

    # Merge weights and actual returns for the same permnos (stocks) based on year-month
    merged_data = pd.merge(weights, returns, on=['permno', 'date'])
    # print(merged_data)
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
print(cumulative_df)

# Graphing prep
market_data = pd.read_csv('mkt_ind.csv')
market_data['date'] = pd.to_datetime(market_data[['year', 'month']].assign(day=1))
market_data.sort_values('date', inplace=True)
plotting_data = pd.merge(cumulative_df, market_data[['date', 'rf', 'sp_ret']], on='date', how='inner')
plotting_data = plotting_data[(plotting_data['date'] >= start_date) & (plotting_data['date'] <= end_date)]

# Calculate cumulative returns for the S&P 500
plotting_data['cumulative_sp500'] = (1 + plotting_data['sp_ret']).cumprod()

# Plot the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(plotting_data['date'], plotting_data['monthly_return'], label='Trading Strategy', color='blue')
plt.plot(plotting_data['date'], plotting_data['cumulative_sp500'], label='S&P 500', color='red')
plt.title('Cumulative Performance of Trading Strategy vs. S&P 500 ' + str(start_date) + ' - ' + str(end_date))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()