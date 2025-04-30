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
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0, 1, 10],
    'reg_alpha': [0, 0.1, 1]
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





def predict_future_usage(model, input_csv, output_csv='future_predictions.csv', plot_results=True):
    """
    Loads a new dataset, preprocesses it, predicts meter readings using the provided model,
    and optionally saves and plots the results.

    Parameters:
        model: Trained XGBoost model (e.g., model or best_model from GridSearchCV)
        input_csv: Path to new data CSV with a 'datetime' column
        output_csv: File to save predictions (default: 'future_predictions.csv')
        plot_results: Whether to display a plot (default: True)
    """

    # Load and preprocess new data
    new_df = pd.read_csv(input_csv)

    if 'datetime' not in new_df.columns:
        raise ValueError("Input CSV must contain a 'datetime' column.")

    new_df['hour'] = pd.to_datetime(new_df['datetime']).dt.hour
    new_df['minute'] = pd.to_datetime(new_df['datetime']).dt.minute
    new_df['day'] = pd.to_datetime(new_df['datetime']).dt.day
    new_df['month'] = pd.to_datetime(new_df['datetime']).dt.month

    new_df.set_index('datetime', inplace=True)

    if 'meter_reading' in new_df.columns:
        new_df.drop('meter_reading', axis=1, inplace=True)

    # Predict
    preds = model.predict(new_df)
    new_df['predicted_meter_reading'] = preds

    # Save to CSV
    new_df.to_csv(output_csv)
    print(f"Predictions saved to {output_csv}")

    # Plot results
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(new_df.index, new_df['predicted_meter_reading'], label='Predicted')
        plt.title('Future Meter Reading Predictions')
        plt.xlabel('Datetime')
        plt.ylabel('Meter Reading')
        plt.legend()
        plt.show()

    return new_df


