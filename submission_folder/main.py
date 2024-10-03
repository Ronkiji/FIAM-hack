import pandas as pd
import lstm
import optimization
import simulation
import portfolio_analysis
import matplotlib.pyplot as plt

# df = pd.read_csv("hackathon_sample_v2.csv")
df = pd.read_csv("testing.csv")

# run the model - all the variables are at the top of the file
results = lstm.run(df.copy())

# if you don't want to run the model
# uncomment line 16, and comment out line 12
# results = pd.read_csv("results.csv")

# Change as needed
start_date = '2010-01-01'
end_date = '2023-12-01'
mkt = pd.read_csv("mkt_ind.csv")
mkt['date'] = pd.to_datetime(mkt[['year', 'month']].assign(day=1))
mkt.sort_values('date', inplace=True)
df = df.loc[:, ['date', 'year', 'month','permno', 'stock_exret']]

weights = optimization.run(results, df)
portfolio_values = simulation.simulate(weights, df)
portfolio_values['date'] = pd.to_datetime(portfolio_values['date'])
# portfolio_analysis.analyze_portfolio(portfolio_values, mkt, weights)

# Graphing prep
plotting_data = pd.merge(portfolio_values, mkt[['date', 'rf', 'sp_ret']], on='date', how='inner')
plotting_data = plotting_data[(plotting_data['date'] >= start_date) & (plotting_data['date'] <= end_date)]

# Calculate cumulative returns for the S&P 500 and our trading strategy
plotting_data['cumulative_portfolio'] = (1 + plotting_data['monthly_return']).cumprod()
plotting_data['cumulative_sp500'] = (1 + plotting_data['sp_ret']).cumprod()

# Plot the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(plotting_data['date'], plotting_data['cumulative_portfolio'], label='Trading Strategy', color='blue')
plt.plot(plotting_data['date'], plotting_data['cumulative_sp500'], label='S&P 500', color='red')
plt.title('Cumulative Performance of Trading Strategy vs. S&P 500 ' + str(start_date) + ' - ' + str(end_date))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

