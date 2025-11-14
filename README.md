# Bajaj_stock_price_analysis_2003to2020
Indian Automotive Giant Bajaj Stock Analysis as well as Forecasting
Bajaj Stock Price Analysis (2003–2020)
This project analyzes 17+ years of Bajaj Auto stock price data. It includes full data cleaning, anomaly detection, forecasting using ARIMA, and machine learning clustering to understand stock behavior and trading patterns.
The goal is to build an end-to-end, real-world data analytics workflow.


Project Overview
This analysis focuses on:

Understanding long-term stock trends

Detecting unusual trading activity

Forecasting closing prices using ARIMA

Grouping similar trading days using K-Means clustering

Applying a consistent data-cleaning and preprocessing pipeline

1. Data Cleaning & Preprocessing

Steps performed:

Standardized column names

Parsed and sorted date column

Converted numerical fields into correct data types

Removed invalid/corrupted rows

Interpolated missing values for continuous time-series

. Anomaly Detection (Z-Score)

Anomalies were detected using a Z-Score threshold of ±3 on:

Close price

Turnover

Traded quantity

Number of trades

3. ARIMA Time-Series Forecasting

Process:

Resampled data to a daily frequency

Performed an 80/20 train-test split

Automatically searched for optimal ARIMA (p, d, q) using AIC

Compared predictions against actual values

Produced a 30-day stock price forecast with confidence intervals

Evaluation metrics used: MAE and RMSE.

4. Visualizations

This project includes:

Long-term closing price trend plots

Price anomaly plots

Forecast vs. actual comparison

Future 30-day prediction chart with uncertainty ban

5. K-Means Clustering Analysis

Features used:

close_price

turnover

total_traded_quantity

no_of_trades

Steps:

Standardized all features

Used Elbow Method to choose optimal cluster count

Applied K-Means (k=3)

Visualized cluster distribution and how clusters change over time

Cluster insights include:

Low-volume stable days

Medium-activity trading days

High-volatility days


