
from typing import Dict
import numpy as np
import cv2


def crop_image(image: np.ndarray, bbox_dict: Dict) -> np.ndarray:

    x_min = int(bbox_dict['bbox']['x_min'])
    y_min = int(bbox_dict['bbox']['y_min'])
    x_max = int(bbox_dict['bbox']['x_max'])
    y_max = int(bbox_dict['bbox']['y_max'])
    crop = image[y_min:y_max, x_min:x_max]

    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop
