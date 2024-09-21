import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

data = pd.read_csv('datasets/20240921_120859_data.csv')
X = data.drop(columns=['output'])
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Option 1: RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Option 2: XGBRegressor
# xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
# xg_reg.fit(X_train, y_train)

# extract feature importances
importances = rf.feature_importances_
# importances = xg_reg.feature_importances_
important_features = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

# select top 20 important features for LSTM
top_features = important_features.head(20).index
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

print(top_features)