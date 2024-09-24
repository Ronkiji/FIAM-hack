import pandas as pd
import random
import os
from datetime import datetime

df = pd.read_csv("hackathon_sample_v2.csv")
output_folder = "datasets"
os.makedirs(output_folder, exist_ok=True)
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# filter df to only include randomly select permnos
unique_permnos = df['permno'].unique()
selected_permnos = random.sample(list(unique_permnos), int(len(unique_permnos) * 0.1))
df_filtered = df[df['permno'].isin(selected_permnos)]

# mapped df
mapping_df = df_filtered[['permno', 'stock_ticker']].drop_duplicates()

# save mapping to txt
mapping_txt_filename = f"{output_folder}/{current_datetime}_mapping.txt"
with open(mapping_txt_filename, "w") as file:
    for index, row in mapping_df.iterrows():
        file.write(f"{row['permno']} = {row['stock_ticker']}\n")

# remove these columns
remove = ["stock_ticker", "cusip", "comp_name", "shrcd", "ret_eom", "exchcd", "size_port"]
df_filtered = df_filtered.drop(columns=[col for col in remove if col in df.columns])

df_filtered = df_filtered.rename(columns={'stock_exret': 'output'})

# save filtered data
filtered_csv_filename = f"{output_folder}/{current_datetime}_data.csv"
df_filtered.to_csv(filtered_csv_filename, index=False)

print(f"Filtered data saved to: {filtered_csv_filename}")
print(f"Mapping saved to: {mapping_txt_filename}")
