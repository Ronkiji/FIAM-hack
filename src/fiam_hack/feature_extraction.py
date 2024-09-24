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

'''
model that trained with data up to end of 2008
2009-01 - stock return -0.1

2009-02
2009 validation 
2010 testing
2011? 
model - 2000-2008

LSTM - 2008 

1 model -> 2000 - 2008
2 model -> 2000 - 2009
LSTM underweighs 2000

stock_ext

prediction S&P 0.2
reality s&p 0.01
prediction ETH 0.2
reality ETH 0.5


INTL 0.3 -> long
APPL 0.1
NVDA -0.1
TSMC -0.5 -> short

TSMC abs(-0.5)

X = volality
S&P +0.2 
S&P -0.2 



'''