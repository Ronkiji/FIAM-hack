import pandas as pd
import lstm
import optimization

# df = pd.read_csv("hackathon_sample_v2.csv")
df = pd.read_csv("testing.csv")

# run the model - all the variables are at the top of the file
results = lstm.run(df.copy())

# if you don't want to run the model
# uncomment line 13, and comment out line 9
# results = pd.read_csv("results.csv")

weights = optimization.run(results, df)

# simulate(weights)

