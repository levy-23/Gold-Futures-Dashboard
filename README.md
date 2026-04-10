# Gold Futures Investment Decision Support Tool

A personalized decision support dashboard for retail investors considering gold futures in their portfolio. 

## What It Does

You input your portfolio, risk tolerance, and investment horizon. The tool runs forecasting, optimization, simulation, and decision analysis to give you a personalized recommendation on gold allocation.


## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. First load takes ~30 seconds (model training), then it's cached.

## Dataset

[Gold Price Forecasting Dataset](https://www.kaggle.com/datasets/vishardmehta/gold-price-forecasting-dataset) — 1,167 trading days of gold futures (GC=F) from June 2021 to January 2026, with OHLCV data and technical indicators.


