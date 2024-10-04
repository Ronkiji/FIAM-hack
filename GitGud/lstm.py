import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from csv_functions import prepare_data, remove_bottom_percentile, keep_top

###############################
# GLOBAL VARIABLE DEFINITIONS #
###############################

# features used for training, retrieved from running feature_selection.py + personal analysis
features = ['mspread', 'rf', 'ebitda_mev', 'ivol_capm_252d', 'prc', 'niq_be', 'rvol_21d',
            'at_me', 'rmax5_rvol_21d', 'z_score', 'seas_2_5an', 'netdebt_me', 'betadown_252d',
            'ret_1_0', 'ncol_gr1a', 'ni_me', 'cash_at', 'prc_highprc_252d', 'dolvol_var_126d', 
            'eps_actual', 'ret_60_12', 'ivol_hxz4_21d', 'seas_1_1an', 'ret_3_1', 'ret_6_1', 
            'bidaskhl_21d', 'betabab_1260d', 'beta_60m']

# final variables
MONTHS = 24
YEARS = 2
EPOCHS = 50
BATCH = 32
TOP = 200

# dates yyyymm format
start_train_date = 200001
end_train_date = 200712
start_val_date = 200801 
end_val_date = 200912
start_oos_date = 201001
end_oos_date = 201012
final_end_date = 202312

# array for all predictions 
predictions = []

# configure logging
logging.basicConfig(
    filename='output/training.log',         # Log file name
    filemode='a',                           # Append mode
    format='%(asctime)s - %(message)s',     # Log format
    level=logging.INFO                      # Set log level
)

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
            # gets past 24 months of data + current
            # for 2010-01, it has all data from 2008-01 till 2010-01
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

def run(df):
    logging.info("LSTM model has started.")

    # prep data
    df = prepare_data(df, features)

    # standardizing all feature columns
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    # standardize the output column
    target_scaler = StandardScaler()
    df[['stock_exret']] = target_scaler.fit_transform(df[['stock_exret']])

    logging.info("Data has been prepped and standardized.\n")

    # rolling window per year
    while end_oos_date <= final_end_date:

        ##########################
        # PREP DATA FOR TRAINING #
        ##########################
        logging.info(f"Training period: {pd.Period(start_train_date, freq='M')} to {pd.Period(end_train_date, freq='M')}")
        logging.info(f"Validation period: {pd.Period(start_val_date, freq='M')} to {pd.Period(end_val_date, freq='M')}")
        logging.info(f"Out-of-sample prediction period: {pd.Period(start_oos_date, freq='M')} to {pd.Period(end_oos_date, freq='M')}")

        # start by retrieving training and validation data, and remove bottom percentile (not peaking into future data)
        tvdf = df[(df['date'] >= str(start_train_date)) & (df['date'] <= str(end_val_date))]
        tvdf = remove_bottom_percentile(tvdf, 50)

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

        ###################################
        # MODEL DEFINITION AND PREDICTION #
        ###################################

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Bidirectional(LSTM(60, return_sequences=True))) 
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(80, activation='relu', return_sequences=False)))
        model.add(Dropout(0.4))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=1.0))
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(X_val, y_val), verbose=0)
        test_predictions = model.predict(X_test)

        # convert the output back using the target_scaler
        y_test = y_test.reshape(-1, 1)
        test_predictions = test_predictions.reshape(-1, 1)
        y_test_original = target_scaler.inverse_transform(y_test).flatten()
        test_predictions_original = target_scaler.inverse_transform(test_predictions).flatten()

        # r2 score calculation
        r2 = r2_score(y_test_original, test_predictions_original)
        logging.info(f"R2 score for out-of-sample predictions: {r2}\n")

        # output
        oos_predictions = pd.DataFrame({
            'date': test_dates,
            'permno': test_permnos,
            'target': y_test_original,
            'predicted': test_predictions_original 
        })

        predictions.append(oos_predictions)
        
        # update dates by one year
        update_dates()

    logging.info("\nModel training and prediction has completed.")

    results = pd.concat(predictions)
   
    # sort by date
    results.sort_values(by=results.columns[1], inplace=True)
    results.to_csv(f'output/model_results.csv', index=False)
    
    # drop target column
    results = results.drop("target", axis=1)
    return results