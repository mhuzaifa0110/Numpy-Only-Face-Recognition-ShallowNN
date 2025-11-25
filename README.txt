Face Detection Neural Network - Assignment 2
=============================================

This project implements a shallow neural network from scratch using only NumPy and Pillow to detect faces.

REQUIREMENTS
------------
- Python 3.7 or higher
- NumPy
- Pillow (PIL)
- Matplotlib (for plotting)

Install dependencies:
  pip install -r requirements.txt

Or install individually:
  pip install numpy pillow matplotlib


FOLDER STRUCTURE
----------------
Assignment 2/
├── dataset/
│   ├── me/                 # Huzaifa's face images (positive samples)
│   └── test/               # Images used by the GUI viewer
├── code/
│   ├── train.py            # Training + evaluation code
│   ├── predict.py          # Prediction helpers / recognizer
│   └── utils.py            # Image loading and preprocessing helpers
├── results/
│   ├── training_loss.png   # Plot of loss vs. epochs
│   └── model_params.npz    # Saved trained weights
├── gui.py                  # Tkinter GUI (viewer + manual evaluation)
├── report/
└── README.txt


SETUP DATASET
-------------
1. Place Huzaifa's face images in: `dataset/me/`
   - Include different angles, lighting, expressions
   - Supported formats: JPG, JPEG, PNG, BMP, GIF, DNG

2. Optional: place test gallery images in `dataset/test/`
   - These are the images you want to browse/annotate in the GUI
   - Supported formats: JPG, JPEG, PNG, BMP, GIF


TRAINING THE MODEL
------------------
Run from project root:

    python main.py --train

This will:
1. Load Huzaifa images from `dataset/me/`
2. Preprocess (grayscale, resize to 720x720, normalize)
3. Split into 60% train / 20% val / 20% test
4. Train the shallow neural network for 200 epochs
5. Save weights to `results/model_params.npz`
6. Plot training/validation loss to `results/training_loss.png`
7. Print evaluation summary to console

Key configuration (edit in `code/train.py`):
- Image size: 720x720 pixels (auto-detected at prediction time)
- Hidden layer: 32 neurons (sigmoid)
- Learning rate: 0.01
- Epochs: 200
- Recognition threshold (global): `RECOGNITION_THRESHOLD = 0.9720`


RUNNING THE GUI
---------------
Launch the Tkinter application:

    python main.py --gui
    # or
    python gui.py

GUI features:
- Shows model info and dataset stats on the left
- Fixed-size image viewer (660×460 canvas) on the right
- Automatic predictions per image with probability display
- Manual evaluation controls (test images only):
  - Select actual label (“Me” / “Not me”)
  - Mark whether the model's prediction was true or false
  - Click “Record Result” to update stats
- Evaluation stats panel (Accuracy, Precision, Recall, False Positive Rate) updates live based on manual annotations
- Custom images loaded from disk do **not** affect the stats


NETWORK ARCHITECTURE
--------------------
- Input Layer: 518,400 neurons (720x720 grayscale image)
- Hidden Layer: 32 neurons with sigmoid activation
- Output Layer: 1 neuron with sigmoid activation (binary classification)

Activation Function: Sigmoid for both layers
Loss Function: Binary Cross-Entropy
Optimization: Gradient Descent with backpropagation


MODEL PARAMETERS
----------------
The trained model is saved in results/model_params.npz containing:
- W1: Weights from input to hidden layer
- b1: Biases for hidden layer
- W2: Weights from hidden to output layer
- b2: Bias for output layer


RESULTS & TROUBLESHOOTING
-------------------------
- `results/training_loss.png`: training and validation loss curves
- `results/model_params.npz`: trained weights (auto-loaded by CLI + GUI)

Common fixes:
1. “No images found” → ensure `dataset/me/` contains supported image files
2. “Model file not found” → run `python main.py --train` first
3. Import errors → run commands from project root, install dependencies
4. Low accuracy → add more positive samples, adjust learning rate/epochs, verify threshold (`RECOGNITION_THRESHOLD`)

NOTES
-----
- Neural network implemented purely with NumPy (no TensorFlow/PyTorch/sklearn)
- Images are auto-resized to match the model's training size (720×720)
- Predictions use the shared threshold constant so CLI, GUI, and training evaluations stay consistent
- Manual evaluation in the GUI is purely user-driven; it doesn't alter the model, only the displayed statistics

