"""
Main entry point for Huzaifa face recognition system.
Run this script to train the model and then recognize test images.
"""

import os
import sys
import argparse
import glob
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path (parent directory of this file)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from code module
from code.train import main as train_main
from code.predict import recognize_huzaifa
import numpy as np


def run_training():
    """
    Run the training process for Huzaifa face recognition.
    """
    train_main()


def run_prediction(image_path=None, model_path=None):
    """
    Run prediction on a single image (for --predict option).
    
    Parameters:
    -----------
    image_path : str
        Path to image file to predict
    model_path : str
        Path to saved model (default: results/model_params.npz)
    """
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(script_dir, "results", "model_params.npz")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("=" * 60)
        print("ERROR: Model not found!")
        print("=" * 60)
        print(f"Model file not found at: {model_path}")
        print("\nPlease train the model first by running:")
        print("  python main.py --train")
        print("or")
        print("  python main.py -t")
        return
    
    # If image path provided, predict on that image
    if image_path:
        if not os.path.exists(image_path):
            print(f"ERROR: Image file not found: {image_path}")
            return
        
        print("=" * 60)
        print("Making Prediction")
        print("=" * 60)
        print(f"Image: {image_path}")
        print(f"Model: {model_path}")
        print("-" * 60)
        
        try:
            # Use Huzaifa recognition function
            result, prob = recognize_huzaifa(image_path, model_path)
            print(f"\nRecognition Result: {result}")
            print(f"Probability: {prob:.4f} ({prob*100:.2f}%)")
            print("=" * 60)
        except Exception as e:
            print(f"ERROR during prediction: {e}")
    else:
        print("=" * 60)
        print("Prediction Mode")
        print("=" * 60)
        print("No image path provided.")


def predict_on_examples(model_path=None):
    """
    Predict on all images in dataset/test folder and display them.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model (default: results/model_params.npz)
    """
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(script_dir, "results", "model_params.npz")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("=" * 60)
        print("ERROR: Model not found!")
        print("=" * 60)
        print(f"Model file not found at: {model_path}")
        return
    
    # Find test folder
    test_folder = os.path.join(script_dir, "dataset", "test")
    
    if not os.path.exists(test_folder):
        print("=" * 60)
        print("Test folder not found!")
        print("=" * 60)
        print(f"Expected folder: {test_folder}")
        print("Please create the folder and add test images.")
        return
    
    # Find all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(test_folder, ext)))
        image_files.extend(glob.glob(os.path.join(test_folder, ext.upper())))
    
    # Remove duplicates (Windows is case-insensitive, so *.jpg and *.JPG match same files)
    image_files = list(set(image_files))
    
    if len(image_files) == 0:
        print("=" * 60)
        print("No images found in test folder!")
        print("=" * 60)
        print(f"Please add images to: {test_folder}")
        return
    
    # Sort files for consistent ordering
    image_files.sort()
    
    print("=" * 60)
    print("Predicting on Test Images")
    print("=" * 60)
    print(f"Found {len(image_files)} images in {test_folder}")
    print("-" * 60)
    
    # Predict on each image and display
    for img_path in image_files:
        try:
            # Get filename without path
            filename = os.path.basename(img_path)
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Use Huzaifa recognition function
            result, prob = recognize_huzaifa(img_path, model_path)
            
            # Print result in binary format
            if "Recognized as Huzaifa" in result:
                print_result = f"{filename_no_ext}: Recognized as Huzaifa"
            else:
                print_result = f"{filename_no_ext}: Unrecognized"
            
            print(print_result)
            
            # Load and display image
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create figure with recognition result
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Display image with proper aspect ratio
            ax.imshow(img, aspect='auto')
            ax.set_title(f"{result}\n(Probability: {prob:.2%})", 
                        fontsize=16, fontweight='bold',
                        color='green' if 'Recognized' in result else 'red',
                        pad=20)
            ax.axis('off')
            
            # Ensure image fits properly in the figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            continue
    
    print("=" * 60)
    print("Prediction complete!")
    print("=" * 60)


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Huzaifa Face Recognition System - Personal Face Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model (default behavior)
  python main.py
  
  # Train explicitly
  python main.py --train
  
  # Predict on a single image
  python main.py --predict path/to/image.jpg
  
  # Launch GUI application
  python main.py --gui
        """
    )
    
    # Create mutually exclusive group for train/predict/gui
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-t', '--train',
        action='store_true',
        help='Train the neural network model'
    )
    group.add_argument(
        '-p', '--predict',
        type=str,
        metavar='IMAGE_PATH',
        help='Predict on an image (provide path to image file)'
    )
    group.add_argument(
        '-g', '--gui',
        action='store_true',
        help='Launch the GUI application'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        type=str,
        metavar='MODEL_PATH',
        default=None,
        help='Path to model file (default: results/model_params.npz)'
    )
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.gui:
        # Launch GUI
        try:
            from gui import main as gui_main
            print("Launching GUI...")
            gui_main()
        except ImportError as e:
            print(f"Error importing GUI: {e}")
            print("Make sure gui.py is in the project root directory.")
        except Exception as e:
            print(f"Error launching GUI: {e}")
    elif args.train:
        run_training()
        # After training, recognize test images
        print("\n" + "=" * 60)
        print("Training complete! Now recognizing test images...")
        print("=" * 60 + "\n")
        predict_on_examples(args.model)
    elif args.predict:
        run_prediction(args.predict, args.model)
    else:
        # No arguments provided: train then recognize test images
        print("Starting...\n")
        run_training()
        # After training, recognize test images
        print("\n" + "=" * 60)
        print("Training complete! Now recognizing test images...")
        print("=" * 60 + "\n")
        predict_on_examples(args.model)


if __name__ == "__main__":
    main()

