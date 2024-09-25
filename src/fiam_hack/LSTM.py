import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, Input
import os
from datetime import datetime
from fiam_hack.evaluate import evaluate

months = 36
years = 3

df = pd.read_csv("datasets/20240924_182044_data.csv")

features = ['mspread', 'rf', 'month', 'prc', 'ret_3_1', 'ret_1_0', 'betadown_252d', 'seas_1_1an', 
                   'ret_6_1', 'seas_2_5an', 'bidaskhl_21d', 'prc_highprc_252d', 'ret_9_1', 'ret_12_7', 
                   'beta_dimson_21d', 'ret_12_1', 'seas_1_1na', 'rvol_21d', 'ivol_capm_252d','market_equity']  

df = df[['date', 'permno', 'output'] + features]
# convert 'yyyymmdd' to 'yyyymm'
df['date'] = df['date'].astype(str).str[:6]
df.ffill(inplace=True)

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

def create_sequences(data, time_steps=months):
    sequences = []
    outputs = []
    
    for permno in data['permno'].unique():
        stock_data = data[data['permno'] == permno]
        # check if the stock has enough data (at least time_steps + 1)
        len(stock_data)
        if len(stock_data) < time_steps + 1:
            continue 
        for i in range(len(stock_data) - time_steps):
            seq = stock_data[features].iloc[i:i + time_steps].values
            output = stock_data['output'].iloc[i + time_steps]
            sequences.append(seq)
            outputs.append(output)
    
    return np.array(sequences), np.array(outputs)

start_train_date = 200001
end_train_date = 200712
start_val_date = 200801 
end_val_date = 200912
start_oos_date = 201001
end_oos_date = 201012

def update_dates():
    global start_train_date, end_train_date, start_val_date, end_val_date, start_oos_date, end_oos_date
    
    start_train_date += 100
    end_train_date += 100
    start_val_date += 100 
    end_val_date += 100
    start_oos_date += 100 
    end_oos_date += 100

all_predictions = []

while end_oos_date <= 202312:

    print(f"Training period: {pd.Period(start_train_date, freq='M')} to {pd.Period(end_train_date, freq='M')}")
    print(f"Validation period: {pd.Period(start_val_date, freq='M')} to {pd.Period(end_val_date, freq='M')}")
    print(f"Out-of-sample prediction period: {pd.Period(start_oos_date, freq='M')} to {pd.Period(end_oos_date, freq='M')}")

    train_data = df[(df['date'] >= str(start_train_date)) & (df['date'] <= str(end_train_date))]
    val_data = df[(df['date'] >= str(start_val_date - (years * 100))) & (df['date'] <= str(end_val_date))]
    test_data = df[(df['date'] >= str(start_oos_date - (years * 100))) & (df['date'] <= str(end_oos_date))]

    # print(f"Number of training data points: {len(train_data)}")
    # print(f"Number of validation data points: {len(val_data)}")
    # print(f"Number of test data points: {len(test_data)}")
    
    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(val_data)
    X_test, y_test = create_sequences(test_data)

    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)

    # define the LSTM model
    model = Sequential()
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Define input layer first
    model.add(Bidirectional(LSTM(250, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    test_predictions = model.predict(X_test)

    r2 = r2_score(y_test, test_predictions.flatten())
    print(f"R² score for out-of-sample predictions: {r2}")

    if len(y_test) > 0:
        # ensure we are slicing to match the length of y_test
        test_dates = test_data['date'].iloc[months:months + len(y_test)].values 
        permnos = test_data['permno'].iloc[months:months + len(y_test)].values
    else:
        print("No valid test data available for creating predictions.")

    oos_predictions = pd.DataFrame({
        'date': test_dates,
        'permno': permnos,
        'target': y_test, 
        'predicted': test_predictions.flatten() 
    })

    all_predictions.append(oos_predictions)

    # save to dump folder
    # oos_predictions.to_csv(os.path.join('dump', f"oos_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"), index=False)
    evaluate(oos_predictions)

    print()

    update_dates()
evaluate(pd.concat(all_predictions))

final_predictions = pd.concat(all_predictions)
final_predictions.reset_index(drop=True, inplace=True)

final_predictions.to_csv(os.path.join('dump', f'final_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)
