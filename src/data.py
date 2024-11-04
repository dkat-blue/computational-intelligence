import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Ініціалізація скейлерів для вхідних і вихідних даних
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

def load_and_preprocess_data(file_path, target_column):
    """
    Завантажити та підготувати дані з файлу CSV.

    Args:
        file_path (str): Шлях до файлу CSV з даними.
        target_column (str): Назва стовпця з цільовими значеннями.

    Returns:
        pd.DataFrame: Підготовлені дані.
    """
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)

    # Поділ на вхідні (X) і цільові (y) змінні
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values.reshape(-1, 1)
    
    # Масштабування вхідних і вихідних даних
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled

def inverse_scale_data(data, is_target=False):
    """
    Інверсія масштабування для вхідних або цільових даних.

    Args:
        data (np.array): Масив даних для інверсії масштабування.
        is_target (bool): Чи є дані цільовими значеннями (True для y, False для X).

    Returns:
        np.array: Дані у вихідному масштабі.
    """
    scaler = scaler_y if is_target else scaler_x
    return scaler.inverse_transform(data)

def split_and_generate_sequences(X, y, look_back=10, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Поділ даних на навчальну, валідаційну та тестову вибірки та створення послідовностей.

    Args:
        X (np.array): Масштабовані вхідні дані.
        y (np.array): Масштабовані цільові дані.
        look_back (int): Кількість попередніх кроків, що використовуються для передбачення.
        batch_size (int): Розмір пакету для генератора.
        val_split (float): Частка валідаційної вибірки.
        test_split (float): Частка тестової вибірки.

    Returns:
        tuple: Генератори для навчальної, валідаційної та тестової вибірок.
    """
    # Розділення даних на навчальну, валідаційну та тестову вибірки
    train_size = int(len(X) * (1 - val_split - test_split))
    val_size = int(len(X) * val_split)
    
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    # Генератори послідовностей
    train_generator = TimeseriesGenerator(X_train, y_train, length=look_back, batch_size=batch_size)
    val_generator = TimeseriesGenerator(X_val, y_val, length=look_back, batch_size=batch_size)
    test_generator = TimeseriesGenerator(X_test, y_test, length=look_back, batch_size=batch_size)

    return train_generator, val_generator, test_generator

def load_data(file_path, target_column, look_back=10, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Завантажити та підготувати дані, створити генератори для послідовностей.

    Args:
        file_path (str): Шлях до файлу CSV з даними.
        target_column (str): Назва стовпця з цільовими значеннями.
        look_back (int): Кількість попередніх кроків для передбачення.
        batch_size (int): Розмір пакету для генератора.
        val_split (float): Частка валідаційної вибірки.
        test_split (float): Частка тестової вибірки.

    Returns:
        tuple: Генератори для навчальної, валідаційної та тестової вибірок.
    """
    X_scaled, y_scaled = load_and_preprocess_data(file_path, target_column)
    return split_and_generate_sequences(X_scaled, y_scaled, look_back, batch_size, val_split, test_split)