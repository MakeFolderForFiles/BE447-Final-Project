from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load and clean data
df = pd.read_csv('trimmed_data_hamilton.csv')
df = df[df['meter_reading'] != 0]

# Ensure datetime is properly parsed and set as index
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df.dropna(subset=['datetime'], inplace=True)
df.set_index('datetime', inplace=True)

df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['day'] = df.index.day
df['month'] = df.index.month
df['weekday'] = df.index.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['lag1'] = df['meter_reading'].shift(1)
df['lag2'] = df['meter_reading'].shift(2)
df['lag3'] = df['meter_reading'].shift(3)
df.dropna(inplace=True)

X = df.drop('meter_reading', axis=1)
y = df['meter_reading']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

y_test = y_test.copy()
y_test.index = X_test.index

params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'min_child_weight': [1,5,10]
}

model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='r2', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

preds = best_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
plt.plot(y_test.index, preds, label='Predicted', alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Meter Readings")
plt.xlabel("Datetime")
plt.ylabel("Meter Reading")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.tight_layout()
plt.show()