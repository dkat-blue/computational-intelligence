from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_simple_ffnn(input_dim):
    """
    Build a simple feedforward neural network with the following architecture:
    - Input Layer: 8 neurons (for the 8 input features)
    - Hidden Layer 1: 128 neurons, ReLU activation
    - Hidden Layer 2: 64 neurons, ReLU activation
    - Hidden Layer 2: 32 neurons, ReLU activation
    - Output Layer: 1 neuron, Linear activation
    Args:
        input_dim (int): Number of input features.
    Returns:
        model (Sequential): Compiled neural network model.
    """
    model = Sequential()
    
    # Input layer and first hidden layer
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    return model