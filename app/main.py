from flask import Flask, request, jsonify
from flask_cors import CORS

import base64

from app.torch_utils import prediction, preprocess_image, pre_image_conv

app = Flask(__name__)

# enable CORS
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/pre_image', methods=['POST'])
def pre_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            
            # preprocess the image, image in bytes
            image = pre_image_conv(img_bytes)
        
            # send the image in base64 format
            image = base64.b64encode(image).decode('utf-8')
        
            return jsonify({'preprocessed_image': image})
        
        except:
            return jsonify({'error': 'error during prediction'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            
            # preprocess the image
            tensor = preprocess_image(img_bytes)
            
            # get the top 5 predictions
            top5_labels, top5 = prediction(tensor)
            data = {'class_labels': top5_labels, 'probabilities': top5}
            
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})