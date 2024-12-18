from flask import Flask, render_template, request, render_template_string
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from io import BytesIO  # For in-memory file handling
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained Keras model
model = load_model('model.h5')

# Check if file extension is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess each digit from the contours
def preprocess_digits_from_contours(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(grey, 75, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate for proper order
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Initialize list for preprocessed digits
    preprocessed_digits = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Crop the digit
        digit = thresh[y:y + h, x:x + w]

        # Resize to 18x18
        resized_digit = cv2.resize(digit, (18, 18))
        # resized_digit = cv2.resize(digit, (28, 28))

        # Pad to 28x28 with black (zeros)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), mode="constant", constant_values=0)

        # Append to list
        preprocessed_digits.append(padded_digit)
        # preprocessed_digits.append(resized_digit)

    return preprocessed_digits, image

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in the request.', 400

    file = request.files['file']

    if file.filename == '':
        return 'No file selected for upload.', 400

    if not allowed_file(file.filename):
        return 'Invalid file type. Please upload a PNG or JPG image.', 400

    try:
        # Save the uploaded file
        
        filename = secure_filename(file.filename)
        if not os.path.isfile(filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

        # Preprocess the image and extract digits
        preprocessed_digits, contoured_image = preprocess_digits_from_contours(file_path)

        # Predict each digit
        predictions = []
        for digit in preprocessed_digits:
            digit_reshaped = digit.reshape(1, 28, 28, 1)
            prediction = model.predict(digit_reshaped)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predictions.append((predicted_digit, confidence))

        # Save the contoured image with bounding boxes
        contoured_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contoured_' + filename)
        cv2.imwrite(contoured_image_path, contoured_image)

        # Render results
        results_html = ''.join(
            f"<p>Digit: {digit}, Confidence: {confidence:.2f}%</p>"
            for digit, confidence in predictions
        )

        return render_template_string(f'''
        <!doctype html>
        <title>Prediction Results</title>
        <h1>Prediction Results</h1>
        <img src="../uploads/contoured_{filename}" alt="Contoured Image" style="max-width: 500px;">
        <h2>Predicted Digits</h2>
        {results_html}
        <a href="/">Try another image</a>
        ''')
    except Exception as e:
        return f"Error processing the image: {e}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return app.send_static_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
