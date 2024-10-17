import numpy as np

# Mean Squared Error Loss Function
def compute_loss(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) loss.
    
    Args:
        y_true (ndarray): True target values (shape: [n_samples, 1])
        y_pred (ndarray): Predicted values from the model (shape: [n_samples, 1])
        
    Returns:
        loss (float): The mean squared error loss.
    """
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    m = y_true.shape[0]
    loss = (1/m) * np.sum((y_true - y_pred) ** 2)
    return loss

# Gradient of the Loss with respect to predictions
def compute_loss_gradient(y_true, y_pred):
    """
    Compute the gradient of the loss with respect to the predicted values.
    
    Args:
        y_true (ndarray): True target values (shape: [n_samples, 1])
        y_pred (ndarray): Predicted values from the model (shape: [n_samples, 1])
        
    Returns:
        dA (ndarray): Gradient of the loss with respect to the predictions (shape: [n_samples, 1])
    """
    m = y_true.shape[0]  # Number of samples
    dA = -(2/m) * (y_true - y_pred)
    return dA

def backpropagate(X, y_true, cache, params):
    """
    Perform backpropagation to compute the gradients for all layers.
    
    Args:
        X (ndarray): Input data (shape: [n_samples, 8])
        y_true (ndarray): True target values (shape: [n_samples, 1])
        cache (dict): Dictionary of intermediate values (Z and A for each layer)
        params (dict): Dictionary of weights and biases for the network
        
    Returns:
        grads (dict): Gradients of the weights and biases for each layer.
    """
    # Initialize a dictionary to hold gradients
    grads = {}
    
    # Number of samples
    m = X.shape[0]
    
    # Ensure all inputs are numpy arrays
    X = np.array(X)
    y_true = np.array(y_true)
    
    # Step 1: Compute the gradient for the output layer (layer 4)
    A4 = np.array(cache['A4'])
    dA4 = compute_loss_gradient(y_true, A4)  # Gradient of loss with respect to A4
    dZ4 = dA4  # Since the activation is linear, dZ4 = dA4
    
    # Gradients for W4 and b4 (output layer)
    A3 = np.array(cache['A3'])
    grads['dW4'] = np.dot(A3.T, dZ4) / m
    grads['db4'] = np.sum(dZ4, axis=0, keepdims=True) / m

    # Step 2: Backpropagate through layer 3 (Hidden Layer 3)
    dA3 = np.dot(dZ4, params['W4'].T)
    dZ3 = dA3 * (np.array(cache['Z3']) > 0)  # ReLU derivative
    
    A2 = np.array(cache['A2'])
    grads['dW3'] = np.dot(A2.T, dZ3) / m
    grads['db3'] = np.sum(dZ3, axis=0, keepdims=True) / m

    # Step 3: Backpropagate through layer 2 (Hidden Layer 2)
    dA2 = np.dot(dZ3, params['W3'].T)
    dZ2 = dA2 * (np.array(cache['Z2']) > 0)  # ReLU derivative
    
    A1 = np.array(cache['A1'])
    grads['dW2'] = np.dot(A1.T, dZ2) / m
    grads['db2'] = np.sum(dZ2, axis=0, keepdims=True) / m

    # Step 4: Backpropagate through layer 1 (Hidden Layer 1)
    dA1 = np.dot(dZ2, params['W2'].T)
    dZ1 = dA1 * (np.array(cache['Z1']) > 0)  # ReLU derivative
    
    grads['dW1'] = np.dot(X.T, dZ1) / m
    grads['db1'] = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return grads

def update_parameters(params, grads, velocities, learning_rate, beta):
    """
    Update parameters using gradient descent with momentum.
    
    Args:
        params (dict): Dictionary of weights and biases.
        grads (dict): Gradients of the weights and biases computed from backpropagation.
        velocities (dict): Dictionary of velocity terms for each weight and bias.
        learning_rate (float): Learning rate for gradient descent.
        beta (float): Momentum factor (between 0 and 1).
    
    Returns:
        params (dict): Updated weights and biases.
        velocities (dict): Updated velocity terms.
    """
    # Update each layer's parameters
    for key in params.keys():
        # Compute the velocity update
        velocities[key] = beta * velocities[key] + (1 - beta) * grads['d' + key]
        
        # Update the weights or biases
        params[key] = params[key] - learning_rate * velocities[key]
    
    return params, velocities

def initialize_velocity(params):
    """
    Initialize the velocity for gradient descent with momentum.
    
    Args:
        params (dict): Dictionary of weights and biases.
    
    Returns:
        velocities (dict): Dictionary of velocity terms initialized to zeros.
    """
    velocities = {}
    
    for key in params.keys():
        velocities[key] = np.zeros_like(params[key])  # Same shape as the weights/biases
    
    return velocities

