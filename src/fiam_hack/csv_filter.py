import pandas as pd
import os

def remove_bottom_percentile(df):

    # Sort the DataFrame by date and market cap
    df.sort_values(['date', 'market_equity'], ascending=[True, False], inplace=True)

    df['percentile'] = df.groupby('date')['market_equity'].transform(lambda x: x.rank(pct=True, ascending=True) * 100)

    permnos = df['permno'].unique()

    for p in permnos:
        stock_data = df[df['permno'] == p]
        if (stock_data['percentile'] < 50).all():
            df = df[df['permno'] != p]

    df.drop('percentile', axis=1, inplace=True)

    return df