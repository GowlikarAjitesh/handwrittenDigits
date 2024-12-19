from flask import Flask, render_template, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Use temporary folder for Render
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_digits_from_contours(image_path):
    image = cv2.imread(image_path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    preprocessed_digits = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        digit = thresh[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), mode="constant", constant_values=0)
        preprocessed_digits.append(padded_digit)

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
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        preprocessed_digits, contoured_image = preprocess_digits_from_contours(file_path)
        predictions = []
        for digit in preprocessed_digits:
            digit_reshaped = digit.reshape(1, 28, 28, 1)
            prediction = model.predict(digit_reshaped)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predictions.append((predicted_digit, confidence))

        contoured_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contoured_' + filename)
        cv2.imwrite(contoured_image_path, contoured_image)

        results_html = ''.join(
            f"<p>Digit: {digit}, Confidence: {confidence:.2f}%</p>"
            for digit, confidence in predictions
        )

        return render_template_string(f'''
        <!doctype html>
        <title>Prediction Results</title>
        <h1>Prediction Results</h1>
        <img src="/uploads/contoured_{filename}" alt="Contoured Image" style="max-width: 500px;">
        <h2>Predicted Digits</h2>
        {results_html}
        <a href="/">Try another image</a>
        ''')
    except Exception as e:
        return f"Error processing the image: {e}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)