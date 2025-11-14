# Bajaj_stock_price_analysis_2003to2020
Bajaj Stock Price Analysis (2003–2020)

This project examines over 17 years of historical stock price data for Bajaj Auto, one of India’s major automotive manufacturers. The analysis covers long-term trend behavior, anomaly detection using statistical techniques, time-series forecasting through ARIMA, and machine-learning clustering to segment trading patterns.

The goal is to build a complete, end-to-end analytics workflow that mirrors how financial analysts, quantitative researchers, and market data teams approach real-world stock behavior.

* Project Overview

This analysis focuses on:

Understanding long-term stock trends

Detecting unusual or outlier trading activity

Forecasting closing prices using ARIMA models

Grouping similar trading behaviors using K-Means clustering

Applying a consistent, structured data-cleaning process

Generating clear visualizations for insight-driven storytelling

* Data Cleaning & Preprocessing

Key steps performed:

Standardized column names for consistency

Parsed and sorted the date column

Converted all numerical fields into correct datatypes

Removed invalid/corrupted rows

Interpolated missing values to maintain a continuous time series

Verified dataset integrity before modeling

This ensures the dataset is reliable for forecasting and machine-learning tasks.

* Anomaly Detection (Z-Score Method)

Anomalies were identified using a Z-Score threshold of ±3, applied on:

Closing price

Turnover

Traded quantity

Number of trades

This method helps highlight trading days with unusually high or low activity compared to historical patterns.

## Anomaly Visual

![image alt](https://github.com/sumanlamichhane984/Bajaj_stock_price_analysis_2002to2020/blob/main/Figure_1.png)

ARIMA Time-Series Forecasting
Process Summary

Resampled data to a consistent daily frequency

Used an 80/20 train–test split

Automatically searched for the best ARIMA (p, d, q) using AIC scoring

Compared model predictions to actual values

Generated a 30-day price forecast with confidence intervals

Evaluated the model using MAE and RMSE

📈 Forecast Plots

(Add your image here)
![image alt](https://github.com/sumanlamichhane984/Bajaj_stock_price_analysis_2002to2020/blob/main/Figure_2.png)

(Add future forecast plot)
![image alt](https://github.com/sumanlamichhane984/Bajaj_stock_price_analysis_2002to2020/blob/main/Figure_3.png)

📊 K-Means Clustering Analysis
Features Used

Close price

Turnover

Total traded quantity

Number of trades

Steps Performed

Standardized all features

Used the Elbow Method to identify the ideal number of clusters

Applied K-Means (k=3)

Visualized cluster segments

Analyzed cluster transitions across different time periods

Cluster Insights

Cluster 1: Low-volume stable trading days

Cluster 2: Regular medium-activity days

Cluster 3: High-volatility, high-turnover days

 Clustering Visual

(Add your image here)
![image alt](https://github.com/sumanlamichhane984/Bajaj_stock_price_analysis_2002to2020/blob/442ffbca8eb9201abcedd4bdfb54aec6f8955612/Figure_5.png)

## Long-Term Trend and Pattern Analysis

The project also includes:

Closing price trend over 17+ years

Moving averages and rolling behavior

Trend shifts around major market events

Volume–price relationship insights

### Trend Chart

(Add your image here)
![image alt](https://github.com/sumanlamichhane984/Bajaj_stock_price_analysis_2002to2020/blob/main/Figure_6.png)
