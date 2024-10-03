import pandas as pd
import lstm

df = pd.read_csv("testing.csv")

# run the model - all the variables are at the top of the file
results = lstm.run(df)

# weights = optimize(results)
# simulate(weights)

