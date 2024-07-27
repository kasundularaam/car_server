from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('arrow_recognition_model.h5')


def predict_arrow(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    prediction = model.predict(img_array)[0]
    classes = ['up', 'down', 'left', 'right']
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return predicted_class, confidence


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "WELCOME"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_path = 'temp_image.jpg'  # Temporary file to save the uploaded image
    image.save(image_path)

    result, confidence = predict_arrow(image_path)

    return jsonify({
        'direction': result,
        'confidence': confidence
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
