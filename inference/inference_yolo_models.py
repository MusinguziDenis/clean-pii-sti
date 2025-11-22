"""Detect bounding boxes of PII in X-ray images."""

import numpy as np
import tqdm
from PIL import Image
from ultralytics import YOLO


def yolo_predict(model:YOLO, img_paths:list, device: str) -> list[dict]:
    """Predicts bounding boxes on images using a YOLO model.

    Args:
        model (nn.Module): YOLO model
        img_paths (List): List of image paths to predict on
        device (str): Device to run the model on
    Returns:
        list[dict]: List of dictionaries containing predictions

    """
    tta_preds = []

    for image_dir in tqdm.tqdm(img_paths, desc="predicting with YOLO"):
        image = Image.open(image_dir).convert("RGB")
        image_array = np.array(image)

        predictions = model(
            image_array, conf=0.4, device=device, verbose=False, augment=False,
            )

        final_predictions = []
        for r in predictions:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                class_name = model.names[cls]

                xmin, ymin, xmax, ymax = xyxy

                final_predictions.append(
                    {
                        "class": class_name,
                        "confidence": conf,
                        "ymin": ymin,
                        "xmin": xmin,
                        "ymax": ymax,
                        "xmax": xmax,
                    },
                )

            if len(boxes) == 0:
                final_predictions.append(
                    {
                        "class": "NEG",
                        "confidence": 0,
                        "ymin": 0,
                        "xmin": 0,
                        "ymax": 0,
                        "xmax": 0,
                    },
                )

        tta_preds += final_predictions

    return tta_preds
