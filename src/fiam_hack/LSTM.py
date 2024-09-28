import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, Input
import os
from datetime import datetime
from fiam_hack.evaluate import evaluate
from fiam_hack.csv_filter import remove_bottom_percentile, keep_top

#######################
# DATA CONFIGURATIONS #
#######################

# generated from csv_transformer.py
df = pd.read_csv("datasets/20240928_012932_data.csv")
print("CSV has been read into df variable.")

# chosen features for ML
features = ['mspread', 'rf', 'month', 'prc', 'ret_3_1', 'ret_1_0', 'betadown_252d', 'seas_1_1an', 
                   'ret_6_1', 'seas_2_5an', 'bidaskhl_21d', 'prc_highprc_252d', 'ret_9_1', 'ret_12_7', 
                   'beta_dimson_21d', 'ret_12_1', 'seas_1_1na', 'rvol_21d', 'ivol_capm_252d']

# reformat the DataFrame 
df = df[['date', 'permno', 'stock_exret', 'market_equity'] + features]

# convert 'yyyymmdd' to 'yyyymm'
df['date'] = df['date'].astype(str).str[:6]

# fill in empty entries
df.ffill(inplace=True)

print("Starting standardization.")

# standardizing all feature columns
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
# RD: Should we standardize the output column too? And convert it back at the end? It changes the scores a lot.

print("Data configuration done.")
print()

########################
# VARIABLE DEFINITIONS #
########################

# final variables
MONTHS = 36 # RD
YEARS = 3 # RD
EPOCHS = 20
BATCH = 20
TOP = 200

# dates yyyymm format
start_train_date = 200001
end_train_date = 200712
start_val_date = 200801 
end_val_date = 200912
start_oos_date = 201001
end_oos_date = 201012

# array for all predictions from 201001 to 202312
predictions = []

#######################
# FUNCTION DEFINITION #
#######################

def create_sequences(data, time_steps=MONTHS, test=False):
    ''' Creates sequences of features fed into the LSTM model '''
    sequences = [] # input sequences
    outputs = [] # matching output (expected)
    dates = [] # dates
    permnos = [] # permnos
    iterations = data['permno'].unique()

    for permno in iterations:
        stock_data = data[data['permno'] == permno]

        # check if the stock has enough data (at least time_steps + 1)
        if len(stock_data) < time_steps + 1:
            continue # skipping
        # iterate over how many month available
        for i in range(len(stock_data) - time_steps):
            # time_steps * num_features
            # gets past 36 months of data + current
            # for 201001, it has all data from 200701 till 201001
            seq = stock_data[features].iloc[i:i + time_steps + 1].values
            output = stock_data['stock_exret'].iloc[i + time_steps]
            sequences.append(seq)
            outputs.append(output)

            if test:
                # get associated stock_data and permno matching the output
                date = stock_data['date'].iloc[i + time_steps]
                pn = stock_data['permno'].iloc[i + time_steps]
                dates.append(date)
                permnos.append(pn)           

    # shape: (num_stocks, time_steps, num_features) (num_stocks * 1)
    # if test, return dates and permnos that matches the output set
    if test:
        return np.array(sequences), np.array(outputs), np.array(dates), np.array(permnos)
    else:
        return np.array(sequences), np.array(outputs)

def update_dates():
    ''' Rolling window: update all the dates by one year '''
    global start_train_date, end_train_date, start_val_date, end_val_date, start_oos_date, end_oos_date
    start_train_date += 100
    end_train_date += 100
    start_val_date += 100 
    end_val_date += 100
    start_oos_date += 100 
    end_oos_date += 100

# rolling window per year
while end_oos_date <= 202312:

    ##########################
    # PREP DATA FOR TRAINING #
    ##########################

    print(f"Training period: {pd.Period(start_train_date, freq='M')} to {pd.Period(end_train_date, freq='M')}")
    print(f"Validation period: {pd.Period(start_val_date, freq='M')} to {pd.Period(end_val_date, freq='M')}")
    print(f"Out-of-sample prediction period: {pd.Period(start_oos_date, freq='M')} to {pd.Period(end_oos_date, freq='M')}")

    # start by retrieving training and validation data, and remove bottom percentile (not peaking into future data)
    tvdf = df[(df['date'] >= str(start_train_date)) & (df['date'] <= str(end_val_date))]
    tvdf = remove_bottom_percentile(tvdf)

    # split it into training, validation, and testing data
    train_data = tvdf[(tvdf['date'] >= str(start_train_date)) & (tvdf['date'] <= str(end_train_date))]
    val_data = tvdf[(tvdf['date'] >= str(start_val_date - (YEARS * 100))) & (tvdf['date'] <= str(end_val_date))]
    test_data = df[(df['date'] >= str(start_oos_date - (YEARS * 100))) & (df['date'] <= str(end_oos_date))]
    # at each month, only keep top TOP stocks by market equity
    # to predict for February, only predict for top x stocks in January (not peaking into future data)
    test_data = keep_top(test_data, TOP)

    # feed time parsed data into create_equences function
    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(val_data)
    X_test, y_test, test_dates, test_permnos = create_sequences(test_data, test=True)

    # DEBUG PRINT STATEMENTS
    # print(f"Number of training data points: {len(train_data)}")
    # print(f"Number of validation data points: {len(val_data)}")
    # print(f"Number of test data points: {len(test_data)}")
    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)

    ###################################
    # MODEL DEFINITION AND PREDICTION #
    ###################################

    # RD: Figure our which layers are good to use, how many units to use on LSTM. 
    # When to use dropout layers? Other layers? 
    # Directional vs Bidirectional? It means if we tune model weights in both directions, or just one
    # Regularization? L1, L2, Early stopping
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(250, return_sequences=True))) 
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150, return_sequences=False))) 
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # RD: are there better loss functions? are there better optimizers?
    model.compile(optimizer='adam', loss='mean_squared_error')
    # RD: epochs and batch are defined in the variable section
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(X_val, y_val), verbose=1)
    # output i dont think is in 1D
    test_predictions = model.predict(X_test)
    # r2 score calculation
    r2 = r2_score(y_test, test_predictions.flatten())
    print(f"R² score for out-of-sample predictions: {r2}")

    # output
    oos_predictions = pd.DataFrame({
        'date': test_dates,
        'permno': test_permnos,
        'target': y_test, 
        'predicted': test_predictions.flatten() 
    })

    predictions.append(oos_predictions)

    # save to dump folder
    oos_predictions.to_csv(os.path.join('dump', f"oos_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"), index=False)
    
    # for curiosity sake
    evaluate(oos_predictions.copy())
    print()
    # update dates by one year
    update_dates()


results = pd.concat(predictions)
evaluate(results.copy())
# sort by date
results.sort_values(by=results.columns[0], inplace=True)
results.to_csv(os.path.join('dump', f'final_output_with_target_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)
# drop target column
results = results.drop("target", axis=1)
results.to_csv(os.path.join('dump', f'final_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)