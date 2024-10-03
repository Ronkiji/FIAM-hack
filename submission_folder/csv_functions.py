import pandas as pd

def prepare_data(df, features):
    '''Prepare data to for the model.'''
    
    # reformat the DataFrame 
    df = df[['date', 'permno', 'stock_exret', 'market_equity'] + features].copy()

    # convert 'yyyymmdd' to 'yyyymm'
    df['date'] = df['date'].astype(str).str[:6]

    # fill in empty entries
    df.ffill(inplace=True)

    return df

def remove_bottom_percentile(df, x):
    '''From the training data, remove stocks that are permanently under x percentile, by market equity.'''
    
    # sort the DataFrame by date and market equity
    df = df.sort_values(['date', 'market_equity'], ascending=[True, False]).copy()

    # create percentile data for each month
    df['percentile'] = df.groupby('date')['market_equity'].transform(lambda x: x.rank(pct=True, ascending=True) * 100)

    for p in df['permno'].unique():
        stock_data = df[df['permno'] == p]
        # if stock is permanently under x percentile for training and validation data, then remove it
        if (stock_data['percentile'] < x).all():
            df = df[df['permno'] != p]

    # delete the added column
    df.drop('percentile', axis=1, inplace=True)

    return df

def keep_top(df, x):
    '''For the testing data, only keep those that in the top x by market equity.'''

    # get year for testing
    test_year = int(df['date'].max()) // 100  

    test_data = []
    permnos = set()

    for month in range(1, 13):
        current_month = test_year * 100 + month  # yyyymm format

        # get top permno for this month
        monthly_data = df[df['date'] == str(current_month)]
        top_permnos = monthly_data.nlargest(x, 'market_equity')['permno']
        
        # get filtered data for this month and add permnos to set
        test_data.append(monthly_data[monthly_data['permno'].isin(top_permnos)])
        permnos.update(top_permnos)

    # filtered data for last year
    filtered_last = pd.concat(test_data, ignore_index=True)
    
    # filtered data for previous years
    filtered_prev = df[df['date'] < str(test_year * 100)]
    filtered_prev = filtered_prev[filtered_prev['permno'].isin(permnos)]
    
    # concatenate all data: filtered_prev + filtered_last
    result_df = pd.concat([filtered_prev, filtered_last], ignore_index=True)
    
    return result_df
