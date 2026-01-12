# Bitcoin Price Prediction – Master’s Thesis Implementation

## Overview
This repository contains the full implementation of my Master’s thesis titled:
**“Bitcoin Price Prediction using Machine Learning and Deep Learning with Behavioral Indicators”**

The project investigates whether behavioral signals (Google Trends) combined with historical price data can improve Bitcoin return prediction and trading performance. The implementation follows a complete end-to-end research pipeline, from data acquisition to statistical and economic evaluation.

---

## Author
**Mohamed Mekky** MSc Data Science, AI and Digital Business  
Student ID: GH1040715  
Date: January 2026

---

## Research Objectives
The thesis is structured around four main research questions:
* **RQ1:** Does a GRU model outperform a Random Forest in directional accuracy?
* **RQ2:** Which model is more stable over time?
* **RQ3:** Are behavioral features (Google Trends) consistently important?
* **RQ4:** Does the model achieve economically meaningful performance (Sharpe Ratio)?

---

## Methodology

### Data Sources
* **Bitcoin historical prices:** Yahoo Finance (BTC-USD).
* **Behavioral data:** Google Trends (multiple CSV files).

### Models
1.  **Random Forest Regressor** (Baseline).
2.  **GRU Neural Network** (Sequence-based Deep Learning).

### Features
* Lagged returns.
* Volatility measures.
* Google Trends level, differences, and moving averages.

---

## Project Structure

```text
.
├── BTC-USD.csv                     # Bitcoin price data
├── multiTimeline*.csv              # Google Trends files
├── thesis_execution.py             # Main implementation code
├── figures/
│   └── bitcoin_prediction_results_FINAL.png  # Visual output
└── README.md                       # Documentation

```

---

## How to Run

### 1. Setup Data

Place all `multiTimeline*.csv` Google Trends files in the project directory.

### 2. Prerequisites

Ensure Python 3.9+ is installed.

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow shap yfinance

```

### 4. Execute

```bash
python thesis_execution.py

```

---

## Output

The script automatically:

* Trains both models (Random Forest & GRU).
* Evaluates statistical significance.
* Computes risk-adjusted performance.
* Generates a final results dashboard saved to: `figures/bitcoin_prediction_results_FINAL.png`

---

## Code Statistics

| Category | Lines |
| --- | --- |
| **Total Lines** | 346 |
| **Source Code** | 249 |
| **Blank Lines** | 68 |
| **Comments** | 29 |

---

## Key Findings (Summary)

* **GRU** achieves higher directional accuracy than Random Forest.
* **GRU** shows stronger temporal stability across quarters.
* **Behavioral (Google Trends)** features are consistently important predictors.
* **GRU trading strategy** exceeds the target Sharpe Ratio threshold.

---

## Notes & License

* **Disclaimer:** This repository is intended for academic purposes only. It does not constitute financial advice.
* **License:** This project is provided for educational and research use only.

```
