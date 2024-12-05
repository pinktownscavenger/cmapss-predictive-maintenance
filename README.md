# Predictive Maintenance Pipeline for IoT Sensor Data

## Overview
This project implements a predictive maintenance pipeline using IoT sensor data from the CMAPSSData dataset. The goal is to predict the Remaining Useful Life (RUL) of engines based on sensor readings, leveraging techniques such as feature correlation analysis, GRU neural networks, and sequence modeling.

## Dataset
The dataset used is **CMAPSSData**, which contains sensor readings from jet engines. Each engine's life cycle is captured until a failure occurs. The main features include:
- **Operational Settings:** Three operating conditions.
- **Sensor Measurements:** Twenty-one sensor readings.
- **RUL:** Remaining Useful Life (calculated from the number of cycles).

**Location:**  
Dataset files are stored in the `data/CMAPSSDATA` directory.

## Features
- Sensor Correlation Analysis: Heatmaps and thresholding to identify critical features.
- Time-Series Modeling: GRU-based deep learning architecture for RUL prediction.
- Preprocessing Pipeline: Scaling, sequence reshaping, and feature selection.
- Evaluation Metrics: Mean Squared Error (MSE) for assessing model performance.

## Exploratory Data Analysis (EDA)

- **Data Cleaning**:  
  - Removed irrelevant columns and computed the Remaining Useful Life (RUL).
  - Handled missing values and outliers.
  
- **Boxplots**:  
  Generated boxplots to visualize the distribution of sensor data across different sensors.

- **Correlation Analysis**:  
  Performed correlation analysis between sensor features and the RUL. Visualized the correlation matrix to identify highly correlated features for further analysis.

- **Feature Selection**:  
  - Used **ExtraTreesRegressor** to determine feature importance.
  - Dropped features with low correlation to RUL (less than 0.5 correlation).
  
- **Model Evaluation**:  
  - Used **Random Forest Regressor** to assess the performance of the feature subsets and calculated **Root Mean Squared Error (RMSE)** for both training and test sets.

The code for EDA can be found in the `src/eda.py` file.

## GRU Model
The project utilizes a Gated Recurrent Unit (GRU) model to predict the Remaining Useful Life (RUL) of engines based on sequential sensor data. GRU is an efficient variant of LSTM (Long Short-Term Memory) that has fewer parameters while still being capable of learning long-term dependencies.

* Architecture: The model consists of multiple GRU layers with dropout for regularization to prevent overfitting.
* Input Sequence: The model processes sequences of sensor readings, where each sequence represents a time window of the engine's operational data.
* Training: The model is trained on the processed and scaled data, and its performance is evaluated using Mean Squared Error (MSE).

## Installation
Clone the repository:
   ```bash
   git clone https://github.com/your-username/cmapss-predictive-maintenance.git
