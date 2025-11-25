"""
Prediction function for face detection.
Loads trained model and makes predictions on new images.
"""

import numpy as np
import os
import sys

# Add project root to path to import utils
# This allows the script to be run from project root: python code/predict.py
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import recognition threshold constant
from code.train import RECOGNITION_THRESHOLD


class ShallowNeuralNetwork:
    """
    Shallow neural network with one hidden layer.
    Same architecture as in train.py for loading saved models.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize with zeros (will be loaded from file)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
    
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    
    def forward_pass(self, X):
        """
        Forward pass through the network.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        Z1 = np.dot(self.W1, X.T) + self.b1
        A1 = self.sigmoid(Z1)
        
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        
        return A2
    
    
    def load_model(self, filepath):
        """
        Load model parameters from file.
        """
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
    
    
    def predict_proba(self, X):
        """
        Predict probabilities for input data.
        """
        A2 = self.forward_pass(X)
        return A2.flatten()
    
    
    def predict(self, X, threshold=None):
        """
        Predict binary class labels.
        """
        if threshold is None:
            threshold = RECOGNITION_THRESHOLD
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def prediction(features, model_path=None, threshold=None):
    """
    Prediction function that takes image features as input and outputs 0 or 1.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Image features (flattened image vector) - can be single sample or batch
        Shape: (n_features,) for single sample or (n_samples, n_features) for batch
    model_path : str
        Path to saved model parameters file (default: results/model_params.npz relative to project root)
    threshold : float, optional
        Classification threshold (default: RECOGNITION_THRESHOLD)
    
    Returns:
    --------
    predictions : numpy.ndarray or int
        Binary predictions (0 or 1)
        Returns int for single sample, array for batch
    """
    # Set default model path if not provided
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, "results", "model_params.npz")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    # Load model first to detect expected input size
    data = np.load(model_path)
    W1 = data['W1']
    model_input_size = W1.shape[1]  # Model expects this many input features
    
    # Determine input size from features
    if features.ndim == 1:
        input_size = features.shape[0]
        features = features.reshape(1, -1)
    else:
        input_size = features.shape[1]
    
    # Check if feature size matches model's expected size
    if input_size != model_input_size:
        import math
        img_dim = int(math.sqrt(model_input_size))
        raise ValueError(
            f"Feature size mismatch: Model expects {model_input_size} features "
            f"({img_dim}x{img_dim} image), but got {input_size} features. "
            f"Please ensure the input features match the model's training size."
        )
    
    # Model parameters (should match training configuration)
    hidden_size = 32
    output_size = 1
    
    # Initialize and load model
    nn = ShallowNeuralNetwork(input_size, hidden_size, output_size)
    nn.load_model(model_path)
    
    # Use default threshold if not provided
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD
    
    # Make prediction
    predictions = nn.predict(features, threshold=threshold)
    
    # Return single value for single sample, array for batch
    if predictions.size == 1:
        return int(predictions[0])
    else:
        return predictions


def load_and_predict_image(image_path, model_path=None):
    """
    Helper function to load an image and make prediction.
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    model_path : str
        Path to saved model parameters file (default: results/model_params.npz)
    
    Returns:
    --------
    prediction : int
        Binary prediction (0 or 1)
    probability : float
        Predicted probability
    """
    from PIL import Image
    
    # Set default model path if not provided
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, "results", "model_params.npz")
    
    # Load model first to detect expected input size
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    data = np.load(model_path)
    W1 = data['W1']
    model_input_size = W1.shape[1]  # Model expects this many input features
    
    # Calculate expected image dimensions from model input size
    import math
    img_dim = int(math.sqrt(model_input_size))
    model_target_size = (img_dim, img_dim)
    
    # Load and preprocess image
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize(model_target_size, Image.Resampling.LANCZOS)  # Resize to match model's expected size
    img_array = np.array(img, dtype=np.float64)
    img_flat = img_array.flatten()
    img_normalized = img_flat / 255.0
    
    # Get prediction (this will check if model exists)
    pred = prediction(img_normalized, model_path)
    
    # Get probability for more detailed output
    input_size = model_input_size
    hidden_size = 32
    output_size = 1
    
    nn = ShallowNeuralNetwork(input_size, hidden_size, output_size)
    nn.load_model(model_path)
    prob = nn.predict_proba(img_normalized.reshape(1, -1))[0]
    
    return pred, prob


def recognize_huzaifa(image_path, model_path=None, recognition_threshold=None):
    """
    Binary face recognition function for Huzaifa.
    Returns one of two categories based on probability:
    - "Recognized as Huzaifa" (probability > threshold)
    - "Unrecognized" (probability <= threshold)
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    model_path : str
        Path to saved model parameters file (default: results/model_params.npz)
    recognition_threshold : float, optional
        Probability threshold above which Huzaifa is recognized (default: RECOGNITION_THRESHOLD)
    
    Returns:
    --------
    recognition_result : str
        Either "Recognized as Huzaifa" or "Unrecognized"
    probability : float
        Predicted probability
    """
    from PIL import Image
    
    # Set default model path if not provided
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, "results", "model_params.npz")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    # Load model first to detect expected input size
    data = np.load(model_path)
    W1 = data['W1']
    model_input_size = W1.shape[1]  # Model expects this many input features
    
    # Calculate expected image dimensions from model input size
    # input_size = width * height, so we can determine the size
    import math
    img_dim = int(math.sqrt(model_input_size))
    model_target_size = (img_dim, img_dim)
    
    # Load and preprocess image (standardize to grayscale, resize, normalize)
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')  # Convert to grayscale
    img = img.resize(model_target_size, Image.Resampling.LANCZOS)  # Resize to match model's expected size
    img_array = np.array(img, dtype=np.float64)
    img_flat = img_array.flatten()
    img_normalized = img_flat / 255.0  # Normalize to [0, 1]
    
    # Verify size matches
    if img_normalized.shape[0] != model_input_size:
        raise ValueError(
            f"Image size mismatch: Model expects {model_input_size} features "
            f"({model_target_size[0]}x{model_target_size[1]}), but got {img_normalized.shape[0]}. "
            f"Please retrain the model with the current image size or use images matching the model's training size."
        )
    
    # Get probability
    input_size = model_input_size
    hidden_size = 32
    output_size = 1
    
    nn = ShallowNeuralNetwork(input_size, hidden_size, output_size)
    nn.load_model(model_path)
    prob = nn.predict_proba(img_normalized.reshape(1, -1))[0]
    
    # Use default threshold if not provided
    if recognition_threshold is None:
        recognition_threshold = RECOGNITION_THRESHOLD
    
    # Binary classification based on probability threshold
    if prob > recognition_threshold:
        result = "Recognized as Huzaifa"
    else:
        result = "Unrecognized"
    
    return result, prob


if __name__ == "__main__":
    """
    Example usage of prediction function.
    """
    # Get default model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "results", "model_params.npz")
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first using train.py")
        print(f"Expected model at: {model_path}")
    else:
        # Example with dummy features (should be replaced with actual image features)
        # In practice, you would load and preprocess an image first
        print("Example usage of prediction() function:")
        print("=" * 50)
        
        # Create dummy features (720x720 grayscale image = 518,400 features)
        dummy_features = np.random.rand(518400)
        
        # Make prediction
        pred = prediction(dummy_features, model_path)
        print(f"Prediction for dummy features: {pred}")
        print("\nTo predict on real images, use:")
        print("  from code.predict import prediction, load_and_predict_image")
        print("  pred, prob = load_and_predict_image('path/to/image.jpg')")