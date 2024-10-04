import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
df = pd.read_csv('src/fiam_hack/results.csv')  # Replace with your file path

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale both 'target' and 'predicted' together
scaled_values = scaler.fit_transform(df[['target', 'predicted']])

# Extract the scaled values
actual_scaled = scaled_values[:, 0]  # First column is the scaled target
predicted_scaled = scaled_values[:, 1]  # Second column is the scaled predicted

# Calculate the R² score for the scaled data
r2 = r2_score(actual_scaled, predicted_scaled)

# Print the R² score
print(f'R² Score: {r2}')
