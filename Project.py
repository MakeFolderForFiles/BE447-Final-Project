from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('uhm2023.csv')

df = df[df['meter_name'].str.contains('admin_serv_1')]


df.drop('stuck', axis=1,inplace=True)
df.drop('upload_time', axis=1,inplace=True)
df.drop('meter_name_old', axis=1,inplace=True)
df.drop('meter_name', axis=1,inplace=True)


df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['minute'] = pd.to_datetime(df['datetime']).dt.minute
df['day'] = pd.to_datetime(df['datetime']).dt.day
df['month'] = pd.to_datetime(df['datetime']).dt.month
df['year'] = pd.to_datetime(df['datetime']).dt.year

df.drop('datetime', axis=1,inplace=True)


#print(df)

X = df.drop('meter_reading', axis=1)
y = df['meter_reading']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = XGBRegressor(objective='reg:squarederror',n_estimators=100, max_depth=5,random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

plt.plot(preds, y_test.index)
plt.show()

df = pd.read_csv('uhm2023.csv')

df = df[df['meter_name'].str.contains('admin_serv_1')]


