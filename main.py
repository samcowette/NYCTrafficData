import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib


# read and clean data, we load in chunks becasue the dataset is 1.7 million rows
df = pd.read_csv('data/Traffic_Volume_2024.csv')
df.dropna(inplace=True)
df = df[['Boro', 'Yr', 'M', 'D', 'HH', 'street', 'Vol', 'Direction']]

# Setup proper date and time values
df['Date'] = pd.to_datetime(df[['Yr', 'M', 'D']].rename(columns={'Yr': 'year', 'M': 'month', 'D': 'day'}))
df['Weekday'] = df['Date'].dt.day_name()
grouped = df.groupby('street')
df['Time'] = pd.to_datetime(df['HH'], format='%H').dt.strftime('%I %p')

# matplotlib
# plot volume by bridge (street)
top_bridge = df.groupby('street')['Vol'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(20, 10))
plt.bar(top_bridge.index, top_bridge.values)
plt.title('Top 10 Streets by Traffic Volume (2024)')
plt.xlabel('Streets')
plt.ylabel('Total Volume')
plt.xticks(rotation=45)
plt.show()

# volume by hour
top_hours = df.groupby('Time')['Vol'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(10,6))
plt.bar(top_hours.index, top_hours.values)
plt.title("Busiest Hours of the Day (2024)")
plt.xlabel('Hour of the Day')
plt.ylabel('Total Volume')
plt.xticks(rotation=0)
plt.ylim(top_hours.values.min() * 0.98, top_hours.values.max() * 1.01)
plt.show()

# format to days of the week by reindexing Weekday (1-7) to names
top_days = df.groupby('Weekday')['Vol'].sum()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
top_days = top_days.reindex(days_order)

# volume by day
plt.figure(figsize=(10,6))
plt.bar(top_days.index, top_days.values)
plt.title('Busiest Days of the Week (2024)')
plt.xlabel('Day of the Week')
plt.ylabel('Total Volume')
plt.xticks(rotation=33)
plt.show()



# setup for machine learning
# We're using Vol as our target variable and day, hour as features
# We will do a 60/40 train/test split
week_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df['weekday_num'] = df['Weekday'].map(week_map)

# These are not good enough, model needs more features (or better ones)
boro_map = {'Manhattan': 0, 'Bronx': 1, 'Brooklyn': 2, 'Queens': 3, 'Staten Island': 4}
df['boro_num'] = df['Boro'].map(boro_map)

# Next let's add streets (should be the best predicter)
# First we have to turn the names into numbers

street_list = df['street'].unique()
street_map = {name: i for i, name in enumerate(street_list)}
df['street_num'] = df['street'].map(street_map)


# Now drop the values we don't need
ml_df = df.dropna(subset=['street_num'])

# Prepare x and y variables for ML
X = ml_df[['weekday_num', 'HH', 'boro_num', 'street_num']]
Y = ml_df['Vol']

# 60/40 train/split model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

# Make Predictions
Y_pred = model.predict(X_test)

# Evaluation of our model using MSE and R2
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Print MSE and r2 for review
print(f"Mean Squared Error: {mse:.2f}") 
print(f"R2 Score: {r2:.2f}")

# Visualing the results
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.title('Actual vs Predicted Traffic Volume')
plt.xlabel('Actual Volume')
plt.ylabel('Predicted Volume')
plt.show()

# Results aren't great, the model cannot handle nonlinear relationships, so we have to try Random Forest
# By creating multiple decision trees, we can capture rush hour, and spikes we see much better

rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
rfModel.fit(X_train, Y_train)

rfY_pred = rfModel.predict(X_test)

print("Mean Squared Error (RF): ", mean_squared_error(Y_test, rfY_pred))
print("R2 Score (RF): ", r2_score(Y_test, rfY_pred))

plt.scatter(Y_test, rfY_pred, alpha=0.5)
plt.title('Actual vs Predicted Traffic Volume (RF)')
plt.xlabel('Actual Volume')
plt.ylabel('Predicted Volume')
plt.show()

# Most important features
importance = rfModel.feature_importances_
for col, score in zip(X.columns, importance):
    print(f"{col}: {score: .4f}")

# Save model
joblib.dump(rfModel, 'NYC_traffic_api/model/traffic_model.pkl')