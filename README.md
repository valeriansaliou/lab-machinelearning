lab-machinelearning
===================

Machine Learning experiments (DQR + predictions) on an example dataset.

# Required environment

 * Python v3.4+
 * PIP

# Install dependencies

`pip install pandas numpy plotly sklearn scipy`

# How to use

This experiment learns from daily and hourly records of bike rentals over 2 years, and tries to predict the amount of bikes rented on a given day (or given hour), knowing the weather and other features.

## 1. Normalize input data

Our algorithms are fed a raw data extract. We need to do some normalization work before we can do any stats on it:

`./1-normalize-data.py`

## 2. Analize data quality

In order to perform a relevant prediction, we need to run a Data Quality Report (aka DQR):

`./2-generate-dqr.py`

## 3. Generate the prediction model

We can now generate the prediction model (from a % of the input data), and test prediction quality (on the remaining % of the input data):

`./3-generate-model.py`

# Preview

**Prediction quality report graph:**

![Prediction Quality Report Graph](https://valeriansaliou.github.io/lab-machinelearning/images/prediction-quality-report-graph.png)

# Copyrights

Refer to the `README.txt` file in the `./data` directory for input data licensing notes.
