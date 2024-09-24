# Re-import necessary libraries after code execution reset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("datasets/20240923_174346_data.csv")

feature_columns = ['mspread', 'rf', 'month', 'ret_3_1', 'ret_1_0', 'betadown_252d', 'seas_1_1an', 
                   'ret_6_1', 'seas_2_5an', 'bidaskhl_21d', 'prc_highprc_252d', 'ret_9_1', 'ret_12_7', 
                   'beta_dimson_21d', 'ret_12_1', 'seas_1_1na', 'year', 'rvol_21d', 'ivol_capm_252d','market_equity']  
target_column = 'output'

# Prepare data for modeling
def prepare_data(df, stock_id, window_size=12):
    """ Prepare data for LSTM with a sliding window approach """
    data = df[df['permno'] == stock_id].sort_values('date')
    features = data[feature_columns].values
    targets = data[target_column].values

    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(targets[i + window_size])
    
    return np.array(X), np.array(y)

# LSTM model definition
def create_lstm_model(input_shape):
    """ Create an LSTM model """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Model training and monthly prediction
def predict_stock_returns(df, start_date, end_date, window_size=12):
    """ Train the LSTM model and predict stock returns month by month """
    unique_stocks = df['permno'].unique()
    predictions = []
    
    # Train on initial data up to start_date
    train_df = df[df['date'] < int(start_date)]

    # Scale the data
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])

    # Prepare training data for all stocks together
    all_X, all_y = [], []
    for stock_id in unique_stocks:
        X, y = prepare_data(train_df, stock_id, window_size)
        all_X.append(X)
        all_y.append(y)
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # Create and train the LSTM model
    input_shape = (all_X.shape[1], all_X.shape[2])
    model = create_lstm_model(input_shape)
    model.fit(all_X, all_y, epochs=5, batch_size=32, verbose=1)

    # Iterate month by month from start_date to end_date
    current_date = pd.to_datetime(start_date)
    while current_date <= pd.to_datetime(end_date):
        # Predict for all stocks in the current month
        current_month_data = df[(df['date'].dt.year == current_date.year) & (df['date'].dt.month == current_date.month)]
        current_month_data[feature_columns] = scaler.transform(current_month_data[feature_columns])
        
        for stock_id in unique_stocks:
            stock_data = current_month_data[current_month_data['permno'] == stock_id]
            if len(stock_data) < window_size:
                continue

            X_test = stock_data[feature_columns].values[-window_size:]
            X_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1]))
            predicted_return = model.predict(X_test)[0][0]
            
            actual_return = stock_data.iloc[-1][target_column]
            predictions.append({
                'permno': stock_id,
                'date': current_date,
                'predicted_return': predicted_return,
                'actual_return': actual_return
            })
        
        # Retune model with data up to the current month
        updated_df = df[df['date'] <= current_date]
        updated_df[feature_columns] = scaler.fit_transform(updated_df[feature_columns])
        
        all_X, all_y = [], []
        for stock_id in unique_stocks:
            X, y = prepare_data(updated_df, stock_id, window_size)
            all_X.append(X)
            all_y.append(y)
        all_X = np.concatenate(all_X, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        
        # Re-train the model
        model.fit(all_X, all_y, epochs=2, batch_size=32, verbose=1)
        
        # Move to the next month
        current_date += pd.DateOffset(months=1)

    return pd.DataFrame(predictions)

# Use the function with your data, start date, and end date
predicted_df = predict_stock_returns(df, start_date='20100131', end_date='2023-12-29')

# Calculate statistics
r2 = r2_score(predicted_df['actual_return'], predicted_df['predicted_return'])
mse = mean_squared_error(predicted_df['actual_return'], predicted_df['predicted_return'])
print(f"R2 Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Print a sample of the predicted vs actual returns
print(predicted_df.head())