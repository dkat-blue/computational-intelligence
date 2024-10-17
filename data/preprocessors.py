import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Standardize the dataset using StandardScaler.
    Args:
        df (pd.DataFrame): Dataset to preprocess.
    Returns:
        pd.DataFrame: Standardized dataset.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)