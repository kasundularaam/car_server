from flask import Flask, request, jsonify
import os
from io import BytesIO
from PIL import Image

import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/upload', methods=['POST'])
def upload():
    raw_data = request.get_data()

    image_stream = BytesIO(raw_data)

    try:
        image = Image.open(image_stream)
        file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        image.save(file_path, 'JPEG')
        arrow_direction = detect_arrows(file_path)
        return jsonify({'direction': arrow_direction}), 200
    except Exception as e:
        return f'An error occurred: {e}', 500


def detect_arrows(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) >= 7:

            arrow_direction = determine_arrow_direction(approx)
            return arrow_direction

    return "no"


def determine_arrow_direction(approx):

    M = cv2.moments(approx)
    if M['m00'] == 0:
        return "Unknown"
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    directions = {"left": 0, "right": 0, "up": 0, "down": 0}

    for point in approx:
        px, py = point[0]
        if px < cx:
            directions["left"] += 1
        elif px > cx:
            directions["right"] += 1
        if py < cy:
            directions["up"] += 1
        elif py > cy:
            directions["down"] += 1

    dominant_direction = max(directions, key=directions.get)
    return dominant_direction


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
