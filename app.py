import os
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model  # Prefer using tensorflow.keras over keras directly
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your model
model = load_model('melanoma_model.h5')

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file to the uploads folder
        file.save(file_path)

        # Preprocess the uploaded image
        try:
            image = Image.open(file_path).convert('RGB')  # Ensure image is in RGB
            image = image.resize((150, 150))  # Resize as per model input
            image = np.array(image)

            if image.shape != (150, 150, 3):  # Check image shape
                return "Error: Invalid image shape. Please upload a 150x150 RGB image."

            image = image.reshape(1, 150, 150, 3)
            image = image / 255.0  # Normalize image

            # Make prediction
            result = model.predict(image)

            # Assuming binary classification
            if result[0][0] > 0.5:
                diagnosis = "Skin cancer detected. Please consult a doctor."
            else:
                diagnosis = "No skin cancer detected. Stay healthy!"

            return render_template('result.html', uploaded_image=filename, diagnosis=diagnosis)
        except Exception as e:
            # Optionally, remove the uploaded file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            return f"An error occurred while processing the image: {str(e)}"
    else:
        return "Error: Unsupported file type. Please upload an image file (png, jpg, jpeg, gif)."

if __name__ == '__main__':
    app.run(debug=True)