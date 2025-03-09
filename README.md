# Wind Turbine Power Prediction

## Overview
This project implements three approaches to predict wind turbine power output based on meteorological time series data:
1. Multilayer Perceptron (MLP) neural network
2. Genetic Algorithm (GA) optimization
3. Hybrid MLP-GA approach

The models use a sliding window technique to capture temporal patterns in weather variables for accurate power output forecasting.

## Technical Details
- **Framework**: TensorFlow + Keras
- **Loss Function**: Mean Absolute Error (MAE)
- **Evaluation Metrics**: 
  - Mean Squared Error (MSE)
  - Coefficient of Determination (R²)
  - Mean Absolute Error (MAE)

## Features
- Time series preprocessing with configurable sliding window
- MLP neural network with optimized architecture
- Genetic Algorithm for feature selection and parameter optimization
- Hybrid approach leveraging GA for MLP weight initialization and architecture search
- Comprehensive evaluation metrics for renewable energy forecasting

## Requirements
- Python 3.8+
- TensorFlow 2.17.0
- Keras 3.6.0
- NumPy 1.26.4
- Pandas 2.2.3
- scikit-learn 1.5.2
- Matplotlib 3.9.2
- Seaborn 0.13.2
- Jupyter notebooks environment

Additional dependencies are listed in `requirements.txt`.
