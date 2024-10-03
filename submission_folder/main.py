import pandas as pd
import lstm
import optimization

df = pd.read_csv("hackathon_sample_v2.csv")

# run the model - all the variables are at the top of the file
# results = lstm.run(df)

test_df = pd.read_csv("final_output_20241001_202812.csv")
weights = optimization.run(results, df)

# simulate(weights)

