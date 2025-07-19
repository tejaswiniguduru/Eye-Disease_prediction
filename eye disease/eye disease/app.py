from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained model
MODEL_PATH = 'model.h5'  # Replace with your trained model path
model = load_model(MODEL_PATH)

# Classes (disease names) - Update as per your dataset
CLASSES = ['Glaucoma', 'Cataract', 'Diabetic Retinopathy', 'Age-Related Macular Degeneration']

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust size as per your model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Prepare image for prediction
    image = prepare_image(filepath)
    predictions = model.predict(image)
    os.remove(filepath)

    disease = CLASSES[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)

    return jsonify({"disease": disease, "confidence": confidence})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
