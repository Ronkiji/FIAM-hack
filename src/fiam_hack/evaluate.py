import pandas as pd

def evaluate(df):
    df.columns = ['date', 'permno', 'col3', 'col4']

    both_positive = (df['col3'] > 0) & (df['col4'] > 0)
    both_negative = (df['col3'] < 0) & (df['col4'] < 0)
    mixed_signs = ~both_positive & ~both_negative

    count_both_positive = both_positive.sum()
    count_both_negative = both_negative.sum()
    count_mixed_signs = mixed_signs.sum()

    total_count = len(df)

    percentage_both_positive = (count_both_positive / total_count) * 100
    percentage_both_negative = (count_both_negative / total_count) * 100
    percentage_combined_both = percentage_both_positive + percentage_both_negative
    percentage_mixed_signs = (count_mixed_signs / total_count) * 100

    print(f"Percentage of entries with both positive or both negative: {percentage_combined_both:.2f}%")
    print(f"Percentage of entries with mixed signs (+- or -+): {percentage_mixed_signs:.2f}%")