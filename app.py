"""Flask app to remove PII from x-ray images."""
from io import BytesIO

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image
from ultralytics import YOLO

from clean.clean import clean_image
from inference.inference_yolo_models import yolo_predict

app = Flask(__name__)

# Load the model once the app starts
model = YOLO("phi_models/best.pt")

@app.route("/predict", methods=["POST"])
def web_clean_image() -> Response | tuple[Response, int]:
    """Clean PII from an image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)

    bboxes = yolo_predict(model, [file.stream], device="cpu")

    image_np = clean_image(image_np, bboxes)

    # Convert the processed image to a format that can be sent in the response
    processed_image = Image.fromarray(image_np)
    buffer = BytesIO()
    processed_image.save(buffer, format="PNG")
    buffer.seek(0)

    return send_file(buffer, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
