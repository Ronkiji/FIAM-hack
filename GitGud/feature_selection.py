import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from matplotlib import pyplot as plt
from fiam_hack.csv_filter import remove_bottom_percentile

# Important columns/factors used not used in the model training
identifying_columns = ['date', 'permno', 'stock_exret']

# Columns/factirs that don't contribute to expected stock return and will be ignored
ignored_columns = ['ret_eom', 'shrcd', 'exchcd', 'year', 'month', 'size_port',  'stock_ticker', 'cusip', 'comp_name']

# Load and reformat csv
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=ignored_columns)
    data['date'] = data['date'].astype(str).str[:6]

    # Fill in black cells
    data.ffill(inplace=True)

    # Reformat dataframe
    columns_to_scale = data.columns.difference(identifying_columns)

    # Standardize feature columns
    scaler = StandardScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    # Retrieve bottom percentile
    remove_bottom_percentile(data)

    return data

# Preprocess data
def preprocess_data(data):  
    try:
        # Select features and target
        X = data.drop(columns=identifying_columns) 
        y = data['stock_exret']  

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        print(f"An error occurred: {e}")
    return X_scaled, y

# Train ensemble model combining XGBoost and Random Forest Model
def train_ensemble_model(X, y):
    # Train Random Forest Model
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    
    # Train XGBoost model
    xgboost = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.001, reg_alpha=0.1, reg_lambda=1.0)

    # Use voting regressor combining XGBoot and Random Forest models
    ensemble = VotingRegressor([('rf', rf), ('xgb', xgboost)])

    # Split data set into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    ensemble.fit(X_train, y_train)
    
    return ensemble

# Plot combined feature importance
def plot_feature_importance(ensemble, feature_names, top_n):
    feature_importances = np.zeros(len(feature_names))
    
    # Aggregate feature importances from both models
    for model_name, model in ensemble.named_estimators_.items():
        if hasattr(model, 'feature_importances_'):
            feature_importances += model.feature_importances_
    
    # Create a dataframe for the combined feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Show top 'n' features
    top_features = feature_importance_df.head(top_n)  
    
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['Feature'], top_features['Importance'], color='blue')
    plt.xlabel("Feature Importance")
    plt.title(f"Top Feature Importances for Predicting Stock Expected Return")
    plt.xticks(rotation=45) 
    plt.savefig(f'output/feature_selection.png')
    plt.show()

def main(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    feature_names = data.drop(columns=identifying_columns).columns
    ensemble = train_ensemble_model(X, y)
    
    plot_feature_importance(ensemble, feature_names, 40)

if __name__ == "__main__":

    csv_file = "csv/hackathon_sample_v2.csv" 
    main(csv_file)