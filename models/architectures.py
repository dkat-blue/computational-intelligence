import numpy as np

# Initialize weights and biases
def initialize_parameters():
    """
    Initialize weights and biases for a network with 8 input neurons, 
    hidden layers of 128, 64, 32 neurons, and 1 output neuron.
    Returns:
        params: Dictionary containing weights and biases for each layer.
    """
    params = {
        'W1': np.random.randn(8, 128) * np.sqrt(2 / 8),
        'b1': np.zeros((1, 128)),
        'W2': np.random.randn(128, 64) * np.sqrt(2 / 128),
        'b2': np.zeros((1, 64)),
        'W3': np.random.randn(64, 32) * np.sqrt(2 / 64),
        'b3': np.zeros((1, 32)),
        'W4': np.random.randn(32, 1) * np.sqrt(2 / 32),
        'b4': np.zeros((1, 1))
    }
    return params

# ReLU activation function
def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

# Forward pass through the network
def forward_pass(X, params):
    """
    Perform forward propagation through the network.
    
    Args:
        X (ndarray): Input data (shape: [n_samples, 8])
        params (dict): Dictionary containing weights and biases.
        
    Returns:
        A4 (ndarray): Final output of the network (shape: [n_samples, 1])
        cache (dict): Dictionary of intermediate values (Z and A for each layer) for use in backprop.
    """
    cache = {}

    # Layer 1: Input -> Hidden (128)
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = leaky_relu(Z1)
    
    # Layer 2: Hidden (128) -> Hidden (64)
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = leaky_relu(Z2)
    
    # Layer 3: Hidden (64) -> Hidden (32)
    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = leaky_relu(Z3)
    
    # Output layer: Hidden (32) -> Output (1)
    Z4 = np.dot(A3, params['W4']) + params['b4']
    A4 = Z4  # Linear activation for output layer
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3, 'Z4': Z4, 'A4': A4}
    return A4, cache