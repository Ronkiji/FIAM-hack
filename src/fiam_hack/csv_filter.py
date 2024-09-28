import pandas as pd
import numpy as np

# stock_ticker and market_equity

df = pd.read_csv("hackathon_sample_v2.csv")

# Convert 'date' from 'yyyymmdd' to 'yyyymm' by slicing the string
df['date'] = df['date'].astype(str).str[:6]

# Sort the DataFrame by date and market cap
df.sort_values(['date', 'market_equity'], ascending=[True, False], inplace=True)
    # print(df)

# Step 1: Calculate the percentile rank for each stock's market cap per month
df['percentile'] = df.groupby('date')['market_equity'].transform(lambda x: x.rank(pct=True, ascending=True) * 100)

#using this to debug to see the percentiles for eachs stock in each month
    # print(df[['date', 'stock_ticker', 'percentile']])

# Step 2: Flag stocks in the bottom 30% each month
df['bottom_30'] = df['percentile'] <= 30 # Gives false for >30 and True for <=30
    # print(df[['date', 'stock_ticker', 'percentile','bottom_30']])

# Function to generate a list of months between two dates in 'yyyymm' format.
def generate_month_list(start_date, end_date):
    """

    I don't think we need this since we already know all the months.    

    Chatgpt made this
    
    """
    start_year = int(start_date[:4])
    start_month = int(start_date[2:])
    end_year = int(end_date[:4])
    end_month = int(end_date[2:]) # 2 because month is only 2 digits
    months = []
    current_year = start_year
    current_month = start_month
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        months.append(f"{current_year:04d}{current_month:02d}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    return months

# Step 3: For each stock, check if it remains in the bottom 30% for the next 12 months
def check_future(group):
    group = group.sort_values('date')
    group['remain_bottom_30'] = False # the stocks that are below 30 percentile

    # I guess we need this? I feel like we can just get the dates on our own but they just guarantee that we don't miss one
    all_dates = generate_month_list(group['date'].min(), group['date'].max()) 

    date_indices = {date: idx for idx, date in enumerate(all_dates)}
    available_dates = set(group['date'])
    for idx, row in group.iterrows():
        if row['bottom_30']:
            current_date = row['date']
            if current_date in date_indices:
                current_index = date_indices[current_date]
                # Get the next 12 months
                future_indices = range(current_index + 1, current_index + 13)
                future_dates = [all_dates[i] for i in future_indices if i < len(all_dates)]
                # Only consider future dates that are available in the data
                future_dates = [date for date in future_dates if date in available_dates]
                future_data = group[group['date'].isin(future_dates)]
                # Check if it ever moves above 30th percentile
                if not future_data.empty and future_data['percentile'].max() <= 30:
                    group.at[idx, 'remain_bottom_30'] = True
    return group

# Apply the function to each stock
df = df.groupby('stock_ticker', group_keys=False).apply(check_future)

# Step 4: Filter out the stocks that remain in the bottom 30% for the next 12 months
df_filtered = df[~df['remain_bottom_30']]

# Output the filtered DataFrame to a new CSV file
df_filtered.to_csv('filtered_data.csv', index=False)