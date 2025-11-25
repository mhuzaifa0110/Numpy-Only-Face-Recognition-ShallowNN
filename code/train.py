"""
Training script for shallow neural network face detection.
Implements neural network from scratch using only NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path to import utils
# This allows the script to be run from project root: python code/train.py
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.utils import load_dataset, split_dataset

# Recognition threshold constant - change this value to adjust classification threshold
RECOGNITION_THRESHOLD = 0.9750


class ShallowNeuralNetwork:
    """
    Shallow neural network with one hidden layer.
    Architecture: Input -> Hidden -> Output
    Activation: Sigmoid for both layers
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (flattened image size)
        hidden_size : int
            Number of neurons in hidden layer
        output_size : int
            Number of output neurons (1 for binary classification)
        learning_rate : float
            Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # W1: weights from input to hidden layer
        # b1: biases for hidden layer
        # W2: weights from hidden to output layer
        # b2: bias for output layer
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))
    
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        σ(x) = 1 / (1 + exp(-x))
        """
        # Clip x to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid function.
        σ'(x) = σ(x) * (1 - σ(x))
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    
    def forward_pass(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (n_samples, n_features)
        
        Returns:
        --------
        Z1 : hidden layer pre-activation
        A1 : hidden layer activation
        Z2 : output layer pre-activation
        A2 : output layer activation (predictions)
        """
        # Reshape X if needed (ensure it's 2D)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Input to hidden layer
        # Z1 = W1 * X^T + b1
        Z1 = np.dot(self.W1, X.T) + self.b1
        A1 = self.sigmoid(Z1)
        
        # Hidden to output layer
        # Z2 = W2 * A1 + b2
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        
        return Z1, A1, Z2, A2
    
    
    def backward_pass(self, X, y, Z1, A1, Z2, A2):
        """
        Backward pass (backpropagation) to compute gradients.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (n_samples, n_features)
        y : numpy.ndarray
            True labels (n_samples,)
        Z1 : numpy.ndarray
            Hidden layer pre-activation
        A1 : numpy.ndarray
            Hidden layer activation
        Z2 : numpy.ndarray
            Output layer pre-activation
        A2 : numpy.ndarray
            Output layer activation (predictions)
        
        Returns:
        --------
        dW1, db1, dW2, db2 : gradients for weight updates
        """
        m = X.shape[0]  # Number of samples
        
        # Reshape y if needed
        if y.ndim == 1:
            y = y.reshape(1, -1)
        else:
            y = y.T
        
        # Output layer error
        # dZ2 = A2 - y (derivative of binary cross-entropy loss with sigmoid)
        dZ2 = A2 - y
        
        # Gradients for output layer
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Hidden layer error
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.sigmoid_derivative(Z1)
        
        # Gradients for hidden layer
        dW1 = (1 / m) * np.dot(dZ1, X)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    
    def update_weights(self, dW1, db1, dW2, db2):
        """
        Update weights using gradient descent.
        
        Parameters:
        -----------
        dW1, db1, dW2, db2 : gradients
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted probabilities
        
        Returns:
        --------
        loss : float
            Mean loss value
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Reshape if needed
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        else:
            y_true = y_true.T
        
        m = y_true.shape[1]
        
        # Binary cross-entropy loss
        loss = -(1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return loss
    
    
    def compute_error(self, y_true, y_pred):
        """
        Compute mean absolute error.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted probabilities
        
        Returns:
        --------
        error : float
            Mean absolute error
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        error = np.mean(np.abs(y_true - y_pred))
        return error
    
    
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, verbose=True):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation labels
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        train_losses : list
            Training loss per epoch
        val_losses : list
            Validation loss per epoch
        """
        train_losses = []
        val_losses = []
        
        if verbose:
            print(f"\nStarting training for {epochs} epochs...")
            print("Progress will be shown every 10 epochs.\n")
        
        for epoch in range(epochs):
            # Forward pass
            Z1, A1, Z2, A2 = self.forward_pass(X_train)
            
            # Compute loss
            train_loss = self.compute_loss(y_train, A2)
            train_losses.append(train_loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward_pass(X_train, y_train, Z1, A1, Z2, A2)
            
            # Update weights
            self.update_weights(dW1, db1, dW2, db2)
            
            # Validation loss
            _, _, _, A2_val = self.forward_pass(X_val)
            val_loss = self.compute_loss(y_val, A2_val)
            val_losses.append(val_loss)
            
            # Print progress every 10 epochs
            if verbose:
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                elif (epoch + 1) == epochs:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if verbose:
            print("\nTraining completed!\n")
        
        return train_losses, val_losses
    
    
    def predict_proba(self, X):
        """
        Predict probabilities for input data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        
        Returns:
        --------
        probabilities : numpy.ndarray
            Predicted probabilities
        """
        _, _, _, A2 = self.forward_pass(X)
        return A2.flatten()
    
    
    def predict(self, X, threshold=None):
        """
        Predict binary class labels.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        threshold : float, optional
            Classification threshold (default: RECOGNITION_THRESHOLD)
        
        Returns:
        --------
        predictions : numpy.ndarray
            Binary predictions (0 or 1)
        """
        if threshold is None:
            threshold = RECOGNITION_THRESHOLD
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    
    def save_model(self, filepath):
        """
        Save model parameters to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Model saved to {filepath}")
    
    
    def load_model(self, filepath):
        """
        Load model parameters from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Model loaded from {filepath}")


def main():
    """
    Main training function.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get the project root directory (parent of code directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration
    faces_folder = os.path.join(project_root, "dataset", "me")
    target_size = (720, 720)  # Image size
    hidden_size = 32  # Number of neurons in hidden layer
    learning_rate = 0.01
    epochs = 200
    
    # Output paths
    results_dir = os.path.join(project_root, "results")
    model_path = os.path.join(results_dir, "model_params.npz")
    plot_path = os.path.join(results_dir, "training_loss.png")
    errors_path = os.path.join(results_dir, "errors.txt")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 50)
    print("Huzaifa Face Recognition System - Training")
    print("=" * 50)
    print("Training model to recognize Huzaifa's face")
    print("-" * 50)
    
    # Load dataset (only from "me" folder)
    print("Loading dataset...")
    X, y = load_dataset(faces_folder, None, target_size=target_size)
    
    if len(X) == 0:
        print("ERROR: No images found in dataset folder!")
        print("Please add images to:")
        print("  - dataset/me/ (Huzaifa's photos)")
        return
    
    # Split dataset
    print("\nSplitting dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    # Initialize network
    input_size = X_train.shape[1]  # Flattened image size
    output_size = 1  # Binary classification
    
    print(f"\nInitializing neural network...")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Output size: {output_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {epochs}")
    
    nn = ShallowNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    # Train network
    print(f"\n{'='*50}")
    print("Starting Training Process")
    print(f"{'='*50}")
    train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=True)
    
    # Save model
    print(f"\nSaving model...")
    nn.save_model(model_path)
    
    # Plot training loss
    print(f"\nPlotting training loss...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    # Evaluate on all sets
    print(f"\nEvaluating model...")
    
    # Training set
    y_train_pred = nn.predict_proba(X_train)
    train_error = nn.compute_error(y_train, y_train_pred)
    train_accuracy = np.mean((y_train_pred >= RECOGNITION_THRESHOLD).astype(int) == y_train)
    
    # Validation set
    y_val_pred = nn.predict_proba(X_val)
    val_error = nn.compute_error(y_val, y_val_pred)
    val_accuracy = np.mean((y_val_pred >= RECOGNITION_THRESHOLD).astype(int) == y_val)
    
    # Test set
    y_test_pred = nn.predict_proba(X_test)
    test_error = nn.compute_error(y_test, y_test_pred)
    test_accuracy = np.mean((y_test_pred >= RECOGNITION_THRESHOLD).astype(int) == y_test)
    
    # Print results
    print(f"\nResults:")
    print(f"  Training - Error: {train_error:.4f}, Accuracy: {train_accuracy:.4f}")
    print(f"  Validation - Error: {val_error:.4f}, Accuracy: {val_accuracy:.4f}")
    print(f"  Test - Error: {test_error:.4f}, Accuracy: {test_accuracy:.4f}")
    
    # Save errors to file
    with open(errors_path, 'w') as f:
        f.write("Mean Errors and Accuracy\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Set:\n")
        f.write(f"  Mean Error: {train_error:.6f}\n")
        f.write(f"  Accuracy: {train_accuracy:.6f}\n\n")
        f.write(f"Validation Set:\n")
        f.write(f"  Mean Error: {val_error:.6f}\n")
        f.write(f"  Accuracy: {val_accuracy:.6f}\n\n")
        f.write(f"Test Set:\n")
        f.write(f"  Mean Error: {test_error:.6f}\n")
        f.write(f"  Accuracy: {test_accuracy:.6f}\n")
    
    print(f"\nErrors saved to {errors_path}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

