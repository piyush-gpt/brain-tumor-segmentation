from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from matplotlib import pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = './static/uploads/'
RESULT_FOLDER = './static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the trained model (saved as .keras format)
model = tf.keras.models.load_model('brain_tumor.keras', custom_objects={'dice_coefficient': lambda y_true, y_pred: y_pred})  # Add custom metrics if necessary

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Load and preprocess the image (resize and normalize)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize to model's input shape
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
    return image

# Function to create color overlay
def create_color_overlay(image_path, pred_mask):
    original_image = cv2.imread(image_path)  # Load original image in color (BGR format)
    pred_mask_resized = cv2.resize(pred_mask.squeeze(), (original_image.shape[1], original_image.shape[0]))
    
    # Convert to RGB if grayscale
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Overlay the tumor region in red
    overlay = original_image.copy()
    overlay[pred_mask_resized == 1] = [0, 0, 255]  # Red in BGR (OpenCV format)
    return overlay

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Preprocess and predict
        input_image = preprocess_image(file_path)
        pred_mask = model.predict(input_image)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize
        
        # Create color overlay
        overlay = create_color_overlay(file_path, pred_mask)
        
        # Save the output image
        result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
        cv2.imwrite(result_path, overlay)
        
        return render_template('index.html', uploaded_image=file.filename, result_image=file.filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
