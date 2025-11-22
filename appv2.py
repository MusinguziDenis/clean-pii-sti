"""Flask app to remove PII from x-ray images."""


from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS, cross_origin
from ultralytics import YOLO

from PIL import Image
import numpy as np
import io
import math

from inference.inference_yolo_models import yolo_predict

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173",
                                                    "http://localhost:4173",
                                                    "http://34.31.76.1:8080",
                                                    "https://dsi.emergentai.ug"]}},
                                                    supports_credentials=True,
                                                    methods=["*"],
                                                    allow_headers=["*"],
                                                )

# Load the model once the app starts
model = YOLO("phi_models/best.pt")

@app.route("/predict", methods=["POST"])
@cross_origin()
def web_clean_image() -> Response | tuple[Response, int]:
    """Detect the location of PII.

    Gets the position of PII and returns a bounding box cordinates of the PII
    in the form [x_min, y_min, x_max, y_max]

    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    bboxes = yolo_predict(model, [file.stream], device="cpu")
    all_bboxes = [([bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]])
                   for bbox in bboxes]
    all_bboxes = [[math.ceil(x) for x in bbox] for bbox in all_bboxes]

    image = Image.open(file)
    image_array = np.array(image)
    for bbox in all_bboxes:
        image_array[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
    
    # Convert numpy array back to image
    clean_image = Image.fromarray(image_array)

    # Return image
    buf = io.BytesIO()
    clean_image.save(buf, format="JPEG")
    buf.seek(0)

    return send_file(buf, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
