import pandas as pd
import numpy as np
import datetime

# Generate mock data
np.random.seed(0)
timestamps = pd.date_range(start='2022-01-01', end='2022-01-10', freq='H')
resource_usage = np.random.randint(1, 100, len(timestamps))


data = pd.DataFrame({'Timestamp': timestamps, 'ResourceUsage': resource_usage})

#it will print the mock data
print(data.head())


data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['HourOfDay'] = data['Timestamp'].dt.hour
data['RollingAverageUsage'] = data['ResourceUsage'].rolling(window=3).mean()  # Example rolling average window of 3 hours


data = data.dropna()


print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


X = data[['DayOfWeek', 'HourOfDay', 'RollingAverageUsage']]
y = data['ResourceUsage']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
from sklearn.ensemble import RandomForestRegressor


rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)

#  Random Forest model performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error (MSE): {mse_rf}")
print(f"Random Forest - Mean Absolute Error (MAE): {mae_rf}")
