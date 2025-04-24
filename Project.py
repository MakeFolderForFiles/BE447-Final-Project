from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv('uhm2023.csv')

# df = df[df['meter_name'].str.contains('admin_serv_1')]


# df.drop('stuck', axis=1,inplace=True)
# df.drop('upload_time', axis=1,inplace=True)
# df.drop('meter_name_old', axis=1,inplace=True)
# df.drop('meter_name', axis=1,inplace=True)

df = pd.read_csv('trimmed_data.csv')



df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['minute'] = pd.to_datetime(df['datetime']).dt.minute
df['day'] = pd.to_datetime(df['datetime']).dt.day
df['month'] = pd.to_datetime(df['datetime']).dt.month
df['weekday'] = pd.to_datetime(df.index).weekday
df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
# df['year'] = pd.to_datetime(df['datetime']).dt.year

df.to_csv('trimmed_data.csv')

#print(df)

df.set_index('datetime', inplace=True)

# temp = df.drop('meter_reading', axis=1)

X = df.drop('meter_reading', axis=1)
y = df['meter_reading']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,shuffle=False)

params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}
model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=params,scoring='r2', cv=3, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)


preds = best_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))


plt.plot(y_test.index, preds)
plt.show()




