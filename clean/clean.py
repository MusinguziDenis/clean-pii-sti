"""clean PII from X-ray images.

Fills the bounding boxes with white pixels.
"""


import numpy as np


def clean_image(image: np.ndarray, boxes: list[dict[str, int]]) -> np.ndarray:
    """Clean PII from an image.

    Args:
        image (np.ndarray): Numpy array of the image
        boxes (List[Dict[str, int]]): List of bounding boxes
    Returns:
        np.ndarray: Processed image

    """
    for box in boxes:
        x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]),\
                                int(box["ymax"])
        image[y1:y2, x1:x2] = 255

    return image
