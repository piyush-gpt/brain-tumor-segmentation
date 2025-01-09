from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
import uuid
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from matplotlib import pyplot as plt

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure upload settings
UPLOAD_FOLDER = './static/uploads/'
RESULT_FOLDER = './static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILES = 10

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model at startup instead of using before_first_request
# In app.py, modify the model loading part
try:
    model = tf.keras.models.load_model('brain_tumor.keras', 
                                     custom_objects={'dice_coefficient': lambda y_true, y_pred: y_pred},
                                     compile=False)  # Add compile=False
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Optionally add fallback loading method if needed
    try:
        model = tf.keras.models.load_model('brain_tumor.keras', compile=False)
    except Exception as e:
        print(f"Fallback loading failed: {str(e)}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(directory):
    files = os.listdir(directory)
    if len(files) > MAX_FILES:
        files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
        for f in files[:-MAX_FILES]:
            try:
                os.remove(os.path.join(directory, f))
            except Exception as e:
                print(f"Error removing file {f}: {str(e)}")

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to load image")
        
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = np.expand_dims(image, axis=(0, -1))
        return image
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def create_color_overlay(image_path, pred_mask):
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError("Failed to load image for overlay")
            
        pred_mask_resized = cv2.resize(pred_mask.squeeze(), 
                                     (original_image.shape[1], original_image.shape[0]))
        
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        overlay = original_image.copy()
        overlay[pred_mask_resized == 1] = [0, 0, 255]
        return overlay
    except Exception as e:
        raise ValueError(f"Error creating overlay: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PNG or JPEG image')
            return redirect(request.url)
        
        # Create secure filename with UUID
        filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save uploaded file
        file.save(file_path)
        cleanup_old_files(UPLOAD_FOLDER)
        
        # Process image and make prediction
        input_image = preprocess_image(file_path)
        pred_mask = model.predict(input_image)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # Create and save overlay
        overlay = create_color_overlay(file_path, pred_mask)
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(result_path, overlay)
        cleanup_old_files(RESULT_FOLDER)
        
        return render_template('index.html', 
                             uploaded_image=filename, 
                             result_image=filename)

    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(request.url)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)