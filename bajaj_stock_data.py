# ======================================================
# 📊 BAJAJ SALES DATA ANALYSIS: CLEANING + ANOMALY + ARIMA FORECAST
# ======================================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ======================================================
# 🔹 STEP 1: Load Data
# ======================================================
df = pd.read_csv(r"C:\Users\sabin\OneDrive\Desktop\Python\data for python\bajaj-2003-2020.csv")

# 🔹 STEP 2: Clean Column Names
# ======================================================
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)

# ======================================================
# 🔹 STEP 3: Parse Dates and Sort
# ======================================================
if 'date' not in df.columns:
    raise KeyError(f"No 'date' column found. Columns available: {df.columns.tolist()}")

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date')

# ======================================================
# 🔹 STEP 4: Convert Numeric Columns
# ======================================================
numeric_cols = ['close_price', 'turnover', 'total_traded_quantity', 'no_of_trades']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=[c for c in numeric_cols if c in df.columns], how='all', inplace=True)

# ======================================================
# 🔹 STEP 5: Detect Anomalies Using Z-Score
# ======================================================
for col in numeric_cols:
    if col in df.columns:
        df[f'z_{col}'] = stats.zscore(df[col], nan_policy='omit')

threshold = 3
df['anomaly'] = np.where(
    ((df.filter(like='z_').abs() > threshold).any(axis=1)),
    'Yes', 'No'
)

# Print anomaly counts
print("🔍 Anomaly Counts:")
print(df['anomaly'].value_counts())

# ======================================================
# 🔹 STEP 5a: Display Anomaly Table
# ======================================================
anomaly_columns = ['date', 'close_price', 'turnover', 'total_traded_quantity', 'no_of_trades'] + [f'z_{col}' for col in numeric_cols if f'z_{col}' in df.columns]
anomaly_data = df[df['anomaly'] == 'Yes'][anomaly_columns]

print("\n⚠️ Detected Anomalies (Table View):")
print(anomaly_data.reset_index(drop=True).to_string(index=False))

# ======================================================
# 🔹 STEP 6: Plot Close Price + Anomalies
# ======================================================
if 'close_price' in df.columns:
    plt.figure(figsize=(12,6))
    plt.plot(df['date'], df['close_price'], label='Close Price', color='blue', linewidth=1)
    plt.scatter(df[df['anomaly']=='Yes']['date'],
                df[df['anomaly']=='Yes']['close_price'],
                color='red', label='Anomalies', s=40)
    plt.title('📈 Close Price with Detected Anomalies', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ======================================================
# 🔹 STEP 7: Prepare Time Series
# ======================================================
df_ts = df[['date', 'close_price']].dropna()
df_ts = df_ts.groupby('date').agg({'close_price': 'last'}).reset_index()
df_ts.set_index('date', inplace=True)

# Daily frequency and interpolation
ts = df_ts['close_price'].asfreq('D').interpolate()

print(f"\n✅ Time Series Range: {ts.index.min()} → {ts.index.max()}")
print(f"Total Length: {len(ts)}")

# ======================================================
# 🔹 STEP 8: Train-Test Split
# ======================================================
train_size = int(len(ts) * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

# ======================================================
# 🔹 STEP 9: Find Best ARIMA Parameters Automatically
# ======================================================
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

best_aic = np.inf
best_params = None
best_model = None

for param in pdq:
    try:
        model = ARIMA(train, order=param)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = param
            best_model = results
    except:
        continue

print(f"\n✅ Best ARIMA parameters: {best_params} (AIC={best_aic:.2f})")

# ======================================================
# 🔹 STEP 10: Forecast & Evaluate
# ======================================================
forecast = best_model.get_forecast(steps=len(test))
predictions = forecast.predicted_mean

mae = mean_absolute_error(test, predictions)
rmse = mean_squared_error(test, predictions) ** 0.5
print(f"\n📊 Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# ======================================================
# 🔹 STEP 11: Visualize Forecast vs Actual
# ======================================================
plt.figure(figsize=(12,5))
plt.plot(train.index, train, label='Train Data', color='blue')
plt.plot(test.index, test, label='Actual Test Data', color='green')
plt.plot(test.index, predictions, label='Forecast', color='red')
plt.title("ARIMA Forecast vs Actual - Bajaj Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================
# 🔹 STEP 12: 30-Day Future Forecast
# ======================================================
final_model = ARIMA(ts, order=best_params)
final_fit = final_model.fit()

future_forecast = final_fit.get_forecast(steps=30)
forecast_mean = future_forecast.predicted_mean
forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
forecast_series = pd.Series(forecast_mean.values, index=forecast_index)

print("\n📅 Next 30-Day Forecast:")
print(forecast_series)

# ======================================================
# 🔹 STEP 13: Visualize 30-Day Forecast with Confidence Intervals
# ======================================================
conf_int = future_forecast.conf_int()
lower_series = pd.Series(conf_int.iloc[:, 0].values, index=forecast_index)
upper_series = pd.Series(conf_int.iloc[:, 1].values, index=forecast_index)

plt.figure(figsize=(12,5))
plt.plot(ts, label='Historical Data', color='blue')
plt.plot(forecast_series, label='30-Day Forecast', color='red')
plt.fill_between(forecast_index, lower_series, upper_series, color='pink', alpha=0.3)
plt.title("📈 30-Day Forecast (ARIMA) - Bajaj Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()


# ======================================================
# 📊 BAJAJ SALES DATA ANALYSIS: K-MEANS CLUSTERING
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# ======================================================
# 🔹 STEP 1: Load Data
# ======================================================


# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)

# Parse date
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date')

# ======================================================
# 🔹 STEP 2: Select numeric features for clustering
# ======================================================
features = ['close_price', 'turnover', 'total_traded_quantity', 'no_of_trades']
df_numeric = df[features].copy()

# Drop rows with all NaNs in features
df_numeric.dropna(how='all', inplace=True)

# Fill remaining NaNs with median
df_numeric.fillna(df_numeric.median(), inplace=True)

# ======================================================
# 🔹 STEP 3: Standardize the data
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================================================
# 🔹 STEP 4: Determine optimal number of clusters (Elbow method)
# ======================================================
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 10), sse, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.grid(True)
plt.show()

# ======================================================
# 🔹 STEP 5: Apply K-Means with chosen k
# ======================================================
k = 3  # Choose number of clusters based on elbow method
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# ======================================================
# 🔹 STEP 6: Cluster Centroids
# ======================================================
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
centroids['cluster'] = range(k)
print("\n📊 Cluster Centroids:")
print(centroids)

# ======================================================
# 🔹 STEP 7: Show sample of data with cluster assignments
# ======================================================
print("\n🔹 Sample data with cluster labels:")
print(df[['date'] + features + ['cluster']].head(10))

# ======================================================
# 🔹 STEP 8: Visualize Clusters (Close Price vs Turnover)
# ======================================================
plt.figure(figsize=(10,6))
sns.scatterplot(x='close_price', y='turnover', hue='cluster', palette='Set2', data=df, s=50)
plt.title("K-Means Clusters: Close Price vs Turnover")
plt.xlabel("Close Price")
plt.ylabel("Turnover")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# ======================================================
# 🔹 STEP 9: Time-based Cluster Visualization
# ======================================================
plt.figure(figsize=(12,6))
for cluster in range(k):
    cluster_data = df[df['cluster'] == cluster]
    plt.plot(cluster_data['date'], cluster_data['close_price'], '.', label=f'Cluster {cluster}', alpha=0.6)

plt.title("Time Series Colored by K-Means Cluster")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
