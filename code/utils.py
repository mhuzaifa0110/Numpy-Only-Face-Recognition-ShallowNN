"""
Utility functions for image loading and preprocessing.
Only uses NumPy and Pillow as required.
"""

import numpy as np
from PIL import Image
import os
import glob


def load_images_from_folder(folder_path, label, target_size=(720, 720)):
    """
    Load images from a folder and preprocess them.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing images
    label : int
        Label for these images (1 for faces, 0 for non-faces)
    target_size : tuple
        Target size (width, height) for resizing images
    
    Returns:
    --------
    images : numpy.ndarray
        Array of flattened image vectors (N, height*width)
    labels : numpy.ndarray
        Array of labels (N,)
    """
    images = []
    labels = []
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.dng']
    
    # Collect all image files
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    # Remove duplicates (Windows is case-insensitive, so *.jpg and *.JPG match same files)
    image_files = list(set(image_files))
    
    print(f"Found {len(image_files)} images in {folder_path}")
    print("Loading images...")
    
    # Load and preprocess each image with progress bar
    failed_count = 0
    total = len(image_files)
    
    for idx, img_path in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            
            # Resize to target size
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float64)
            
            # Flatten the image
            img_flat = img_array.flatten()
            
            # Normalize pixel values to [0, 1]
            img_normalized = img_flat / 255.0
            
            images.append(img_normalized)
            labels.append(label)
            
            # Progress bar
            progress = (idx + 1) / total
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            percent = progress * 100
            print(f'\r[{bar}] {percent:.1f}% ({idx + 1}/{total})', end='', flush=True)
            
        except Exception as e:
            failed_count += 1
            filename = os.path.basename(img_path)
            print(f"\nError loading image {filename}: {e}")
            continue
    
    print()  # New line after progress bar
    
    if failed_count > 0:
        print(f"Warning: {failed_count} images failed to load out of {total} found")
        print(f"Successfully loaded: {len(images)} images")
    
    return np.array(images), np.array(labels)


def load_dataset_single_folder(faces_folder, target_size=(720, 720)):
    """
    Load dataset from a single folder (positive examples only).
    Trains only on Huzaifa's images.
    
    Parameters:
    -----------
    faces_folder : str
        Path to folder containing face images (positive examples)
    target_size : tuple
        Target size for resizing images
    
    Returns:
    --------
    X : numpy.ndarray
        Array of image features (N, height*width)
    y : numpy.ndarray
        Array of labels (N,) - all will be 1 (positive)
    """
    # Load face images (label = 1)
    face_images, face_labels = load_images_from_folder(faces_folder, label=1, target_size=target_size)
    
    if len(face_images) == 0:
        return np.array([]), np.array([])
    
    print(f"\nTotal dataset size: {len(face_images)} images")
    print(f"  - All images are positive examples (Huzaifa)")
    
    return face_images, face_labels


def load_dataset(faces_folder, nonfaces_folder=None, target_size=(720, 720)):
    """
    Load the complete dataset from faces and optionally nonfaces folders.
    If nonfaces_folder is None, uses single folder approach (positive examples only).
    
    Parameters:
    -----------
    faces_folder : str
        Path to folder containing face images
    nonfaces_folder : str or None
        Path to folder containing non-face images (None for single folder approach)
    target_size : tuple
        Target size for resizing images
    
    Returns:
    --------
    X : numpy.ndarray
        Array of image features (N, height*width)
    y : numpy.ndarray
        Array of labels (N,)
    """
    if nonfaces_folder is None or not os.path.exists(nonfaces_folder):
        # Use single folder approach (positive examples only)
        return load_dataset_single_folder(faces_folder, target_size)
    
    # Load face images (label = 1)
    face_images, face_labels = load_images_from_folder(faces_folder, label=1, target_size=target_size)
    
    # Load non-face images (label = 0)
    nonface_images, nonface_labels = load_images_from_folder(nonfaces_folder, label=0, target_size=target_size)
    
    # Combine datasets
    X = np.vstack([face_images, nonface_images])
    y = np.hstack([face_labels, nonface_labels])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Total dataset size: {len(X)} images")
    print(f"  - Face images: {np.sum(y == 1)}")
    print(f"  - Non-face images: {np.sum(y == 0)}")
    
    return X, y


def split_dataset(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split dataset into training, validation, and test sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Label vector
    train_ratio : float
        Proportion of data for training (default: 0.6)
    val_ratio : float
        Proportion of data for validation (default: 0.2)
    test_ratio : float
        Proportion of data for testing (default: 0.2)
    
    Returns:
    --------
    X_train, y_train : training set
    X_val, y_val : validation set
    X_test, y_test : test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split indices
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    print(f"  - Test: {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


