import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from matplotlib import pyplot as plt
from fiam_hack.csv_filter import remove_bottom_percentile

identifying_columns = ['date', 'permno', 'stock_exret']
ignored_columns = ['ret_eom', 'shrcd', 'exchcd', 'year', 'month', 'size_port',  'stock_ticker', 'cusip', 'comp_name']

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=ignored_columns)
    data['date'] = data['date'].astype(str).str[:6]
    data.ffill(inplace=True)
    columns_to_scale = data.columns.difference(identifying_columns)
    scaler = StandardScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    remove_bottom_percentile(data)
    
    return data

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

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.001, reg_alpha=0.1, reg_lambda=1.0)
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation scores: {scores}")

    # Train the model
    model.fit(X_train, y_train)
    return model

def plot_feature_importance(model, feature_names, top_n):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort the DataFrame
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    top_features = feature_importance_df.head(top_n)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['Feature'], top_features['Importance'], color='blue')
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importance for Predicting Stock Expected Return")
    plt.xticks(rotation=45) 
    plt.savefig('C:\\Users\\ryan\\FIAM-hack\\feature_importance.png')
    plt.show()


def main(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    
    feature_names = data.drop(columns=identifying_columns).columns
    
    model = train_xgboost(X, y)
    plot_feature_importance(model, feature_names, 40)

if __name__ == "__main__":

    csv_file = "C:\\Users\\ryan\\FIAM-hack\\hackathon_sample_v2.csv" 
    main(csv_file)