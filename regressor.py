import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


df = pd.read_csv(r"C:\Users\HariNarayanan\Downloads\archive (4)\Taxi_Trip_Data_preprocessed.csv")
df = df.sample(n=50000, random_state=42)


df = pd.get_dummies(df, columns=['payment_type'])


X = df.drop(columns=['fare_amount'])
y = df['fare_amount']  # target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

print("Mean absolute error:", mean_absolute_error(y_test, y_pred))


sample_idx = np.random.choice(len(y_test), size=2000, replace=False)
y_test_sample = y_test.iloc[sample_idx]
y_pred_sample = y_pred[sample_idx]

plt.figure(figsize=(8, 8))
plt.scatter(y_test_sample, y_pred_sample, alpha=0.5)
plt.plot([y_test_sample.min(), y_test_sample.max()],
         [y_test_sample.min(), y_test_sample.max()], 'r--')
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare (sample of 2000 points)")
plt.show()

joblib.dump(model, 'taxifare_model.pkl')

print(df.head())
