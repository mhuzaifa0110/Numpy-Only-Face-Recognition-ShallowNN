import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import numpy as np
import glob

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.predict import recognize_huzaifa, ShallowNeuralNetwork


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Huzaifa Face Recognition System")
        self.root.geometry("1200x800")
        
        # Get project paths
        self.project_root = script_dir
        self.model_path = os.path.join(self.project_root, "results", "model_params.npz")
        self.test_folder = os.path.join(self.project_root, "dataset", "test")
        
        # Current image index for test folder navigation
        self.current_image_index = 0
        self.test_images = []
        self.load_test_images()

        # Manual evaluation tracking
        self.annotation_records = {}
        self.current_image_path = None
        self.current_prediction_result = None
        self.annotation_state_enabled = False
        
        # Create GUI components
        self.create_widgets()
        self.load_results()
        
        # Bind arrow keys
        self.root.bind('<Left>', self.previous_image)
        self.root.bind('<Right>', self.next_image)
        self.root.focus_set()
    
    def load_test_images(self):
        """Load all test images from test folder."""
        if not os.path.exists(self.test_folder):
            return
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        for ext in extensions:
            self.test_images.extend(glob.glob(os.path.join(self.test_folder, ext)))
            self.test_images.extend(glob.glob(os.path.join(self.test_folder, ext.upper())))
        
        # Remove duplicates
        self.test_images = list(set(self.test_images))
        self.test_images.sort()
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Results and Errors
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        
        # Results section
        results_label = ttk.Label(left_panel, text="Training Results", font=("Arial", 14, "bold"))
        results_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.results_text = tk.Text(left_panel, width=40, height=15, wrap=tk.WORD, font=("Courier", 10))
        results_scroll = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        left_panel.rowconfigure(1, weight=1)
        
        # Evaluation stats section (fixed box)
        stats_frame = ttk.LabelFrame(left_panel, text="Evaluation Stats", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 5))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.configure(width=400, height=170)
        stats_frame.grid_propagate(False)
        
        self.stats_label = ttk.Label(
            stats_frame,
            text="No evaluations recorded yet.\nSelect options under Manual Evaluation,\nthen click 'Record Result'.",
            font=("Courier", 10),
            justify=tk.LEFT,
            anchor=tk.NW
        )
        self.stats_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        left_panel.rowconfigure(3, weight=1)
        
        # Right panel - Image Viewer
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(3, weight=1)  # Image frame row should expand
        
        # Image viewer title
        viewer_label = ttk.Label(right_panel, text="Image Prediction Viewer", font=("Arial", 14, "bold"))
        viewer_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Prediction info frame (moved above image)
        info_frame = ttk.LabelFrame(right_panel, text="Prediction Results", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)
        
        self.prediction_label = ttk.Label(info_frame, text="No prediction yet", font=("Arial", 12, "bold"))
        self.prediction_label.grid(row=0, column=0, sticky=tk.W)
        
        self.probability_label = ttk.Label(info_frame, text="", font=("Arial", 11))
        self.probability_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        self.image_name_label = ttk.Label(info_frame, text="", font=("Arial", 10), foreground="gray")
        self.image_name_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Manual evaluation frame (for test images only)
        annotation_frame = ttk.LabelFrame(right_panel, text="Manual Evaluation (Test Images Only)", padding="10")
        annotation_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        annotation_frame.columnconfigure(0, weight=1)

        self.actual_var = tk.StringVar()
        self.pred_correct_var = tk.StringVar()
        self.annotation_widgets = []

        actual_frame = ttk.Frame(annotation_frame)
        actual_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Label(actual_frame, text="Actual label:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        rb_me = ttk.Radiobutton(actual_frame, text="Me", variable=self.actual_var, value="me")
        rb_not_me = ttk.Radiobutton(actual_frame, text="Not me", variable=self.actual_var, value="not_me")
        rb_me.grid(row=0, column=1, padx=5)
        rb_not_me.grid(row=0, column=2, padx=5)
        self.annotation_widgets.extend([rb_me, rb_not_me])

        pred_frame = ttk.Frame(annotation_frame)
        pred_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        ttk.Label(pred_frame, text="Prediction evaluation:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        rb_pred_true = ttk.Radiobutton(pred_frame, text="Prediction is correct", variable=self.pred_correct_var, value="true")
        rb_pred_false = ttk.Radiobutton(pred_frame, text="Prediction is incorrect", variable=self.pred_correct_var, value="false")
        rb_pred_true.grid(row=0, column=1, padx=5)
        rb_pred_false.grid(row=0, column=2, padx=5)
        self.annotation_widgets.extend([rb_pred_true, rb_pred_false])

        self.record_button = ttk.Button(annotation_frame, text="Record Result", command=self.record_manual_evaluation)
        self.record_button.grid(row=2, column=0, sticky=tk.E, pady=(10, 0))
        self.annotation_widgets.append(self.record_button)

        # Image display frame with fixed Canvas
        image_frame = ttk.Frame(right_panel, relief=tk.SUNKEN, borderwidth=2, width=720, height=520)
        image_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        image_frame.grid_propagate(False)
        
        self.canvas_width = 660
        self.canvas_height = 460
        self.image_canvas = tk.Canvas(
            image_frame,
            bg="white",
            highlightthickness=0,
            width=self.canvas_width,
            height=self.canvas_height
        )
        self.image_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.image_label = ttk.Label(image_frame, text="No image loaded", anchor=tk.CENTER, font=("Arial", 12))
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.canvas_image_id = None
        
        # Navigation and controls frame
        controls_frame = ttk.Frame(right_panel)
        controls_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
        # Navigation buttons
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(nav_frame, text="◄ Previous", command=self.previous_image).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Next ►", command=self.next_image).grid(row=0, column=1, padx=5)
        
        self.image_counter_label = ttk.Label(nav_frame, text="", font=("Arial", 10))
        self.image_counter_label.grid(row=0, column=2, padx=10)
        
        # File selection button
        ttk.Button(controls_frame, text="Select Image from Device", command=self.select_image).grid(row=0, column=1, padx=10, sticky=tk.E)
        
        # Instructions
        instructions = ttk.Label(right_panel, 
                                text="Use ◄ ► arrow keys or buttons to navigate test images.\nAnnotate ONLY test images, then click 'Record Result' to update stats.",
                                font=("Arial", 9), foreground="gray")
        instructions.grid(row=5, column=0, pady=(10, 0))
        
        # Load first image if available
        if len(self.test_images) > 0:
            self.current_image_index = 0
            self.display_image(self.test_images[0])
    
    def load_results(self):
        """Load and display training results and errors."""
        # Display model info
        results_content = "Model Information\n"
        results_content += "=" * 40 + "\n\n"
        
        if os.path.exists(self.model_path):
            results_content += "✓ Model file found\n"
            results_content += f"Location: {self.model_path}\n\n"
            
            # Try to load model info
            try:
                data = np.load(self.model_path)
                results_content += "Model Parameters:\n"
                results_content += f"  - W1 shape: {data['W1'].shape}\n"
                results_content += f"  - W2 shape: {data['W2'].shape}\n"
                results_content += f"  - Hidden neurons: {data['W1'].shape[0]}\n"
                results_content += f"  - Input size: {data['W1'].shape[1]}\n"
            except:
                results_content += "  (Could not load model details)\n"
        else:
            results_content += "✗ Model file not found\n"
            results_content += "Please train the model first.\n"
        
        results_content += "\n" + "=" * 40 + "\n"
        results_content += f"\nTest Images Found: {len(self.test_images)}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_content)
    
    def display_image(self, image_path):
        """Display an image and make prediction."""
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image not found: {image_path}")
            return
        
        try:
            # Load image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use fixed canvas size
            canvas_width = self.canvas_width
            canvas_height = self.canvas_height
            
            # Calculate resize to fit canvas while maintaining aspect ratio
            img_width, img_height = img.size
            max_display_width = canvas_width - 30  # Padding
            max_display_height = canvas_height - 30  # Padding
            
            scale_w = max_display_width / img_width
            scale_h = max_display_height / img_height
            scale = min(scale_w, scale_h)  # Fit within canvas
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image to fit canvas
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
            
            # Clear previous image from canvas
            if self.canvas_image_id:
                self.image_canvas.delete(self.canvas_image_id)
            
            # Hide the placeholder label
            self.image_label.place_forget()
            
            # Center image on canvas
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Add image to canvas
            self.canvas_image_id = self.image_canvas.create_image(
                center_x, center_y, 
                anchor=tk.CENTER, image=photo
            )
            
            # Keep a reference to prevent garbage collection
            self.image_canvas.image = photo
            
            # Update image name
            image_name = os.path.basename(image_path)
            self.image_name_label.configure(text=f"Image: {image_name}")
            
            # Make prediction
            self.predict_image(image_path)
            
            # Update counter and annotation controls
            self.current_image_path = image_path
            if image_path in self.test_images:
                idx = self.test_images.index(image_path)
                self.current_image_index = idx
                self.image_counter_label.configure(
                    text=f"Image {idx + 1} of {len(self.test_images)}"
                )
                self.set_annotation_controls_state(True)
                # Prefill selections if already annotated
                record = self.annotation_records.get(image_path)
                if record:
                    self.actual_var.set(record["actual"])
                    self.pred_correct_var.set("true" if record["pred_correct"] else "false")
                else:
                    self.actual_var.set("")
                    self.pred_correct_var.set("")
            else:
                self.image_counter_label.configure(text="Custom image")
                self.set_annotation_controls_state(False)
                self.actual_var.set("")
                self.pred_correct_var.set("")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
            self.show_placeholder()
    
    def predict_image(self, image_path):
        """Make prediction on an image."""
        if not os.path.exists(self.model_path):
            self.prediction_label.configure(text="Model not found. Please train first.", foreground="red")
            self.probability_label.configure(text="")
            return
        
        try:
            result, prob = recognize_huzaifa(image_path, self.model_path)
            
            # Update prediction label
            self.prediction_label.configure(text=f"Result: {result}", font=("Arial", 12, "bold"))
            
            # Color code based on result (binary classification)
            if "Recognized as Huzaifa" in result:
                self.prediction_label.configure(foreground="green")
            else:
                self.prediction_label.configure(foreground="red")
            
            # Update probability
            self.probability_label.configure(
                text=f"Probability: {prob:.4f} ({prob*100:.2f}%)",
                font=("Arial", 11)
            )
            self.current_prediction_result = result
        
        except Exception as e:
            self.prediction_label.configure(text=f"Prediction error: {str(e)}", foreground="red")
            self.probability_label.configure(text="")
    
    def next_image(self, event=None):
        """Navigate to next image."""
        if len(self.test_images) == 0:
            messagebox.showinfo("Info", "No test images found in test folder.")
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.test_images)
        self.display_image(self.test_images[self.current_image_index])
    
    def previous_image(self, event=None):
        """Navigate to previous image."""
        if len(self.test_images) == 0:
            messagebox.showinfo("Info", "No test images found in test folder.")
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.test_images)
        self.display_image(self.test_images[self.current_image_index])
    
    def select_image(self):
        """Open file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.display_image(file_path)

    def set_annotation_controls_state(self, enabled):
        state = "normal" if enabled else "disabled"
        for widget in self.annotation_widgets:
            widget.configure(state=state)
        if not enabled:
            self.actual_var.set("")
            self.pred_correct_var.set("")
        self.annotation_state_enabled = enabled

    def show_placeholder(self, message="No image loaded"):
        """Show placeholder text in the image area."""
        if self.canvas_image_id:
            self.image_canvas.delete(self.canvas_image_id)
            self.canvas_image_id = None
        self.image_canvas.image = None
        self.image_label.configure(text=message)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def record_manual_evaluation(self):
        """Record manual evaluation for the current test image."""
        if self.current_image_path not in self.test_images:
            messagebox.showinfo("Info", "Manual evaluation is only available for test images.")
            return

        actual = self.actual_var.get()
        pred_correct = self.pred_correct_var.get()

        if not actual or not pred_correct:
            messagebox.showwarning("Missing Selection", "Please select both the actual label and prediction evaluation.")
            return

        # Store annotation
        self.annotation_records[self.current_image_path] = {
            "actual": actual,
            "pred_correct": pred_correct == "true"
        }

        self.update_stats_display()
        messagebox.showinfo("Recorded", "Evaluation recorded for this image.")

    def update_stats_display(self):
        """Compute and display evaluation statistics."""
        if len(self.annotation_records) == 0:
            self.stats_label.configure(
                text="No evaluations recorded yet.\nSelect options and click 'Record Result'."
            )
            return

        tp = tn = fp = fn = 0
        for record in self.annotation_records.values():
            actual = record["actual"]
            is_correct = record["pred_correct"]

            if actual == "me" and is_correct:
                tp += 1
            elif actual == "me" and not is_correct:
                fn += 1
            elif actual == "not_me" and is_correct:
                tn += 1
            else:
                fp += 1

        total = len(self.annotation_records)

        def safe_ratio(num, denom):
            return (num / denom) * 100 if denom > 0 else None

        accuracy = safe_ratio(tp + tn, total)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        fpr = safe_ratio(fp, fp + tn)

        stats_text = [
            f"Entries: {total}",
            f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}",
            "",
            self.format_stat_line("Accuracy", f"(TP + TN) / Total", tp + tn, total, accuracy),
            self.format_stat_line("Precision", "TP / (TP + FP)", tp, tp + fp, precision),
            self.format_stat_line("Recall", "TP / (TP + FN)", tp, tp + fn, recall),
            self.format_stat_line("False Positive Rate", "FP / (FP + TN)", fp, fp + tn, fpr)
        ]

        self.stats_label.configure(text="\n".join(stats_text))

    def format_stat_line(self, label, formula, numerator, denominator, value):
        if value is None:
            return f"{label} = {formula} = (insufficient data)"
        return (f"{label} = {formula} = ({numerator} / {denominator}) "
                f"= {value:.1f}%")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

