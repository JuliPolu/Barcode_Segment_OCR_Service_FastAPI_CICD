from typing import Dict
import cv2
import numpy as np

import albumentations as albu
import torch
from albumentations.pytorch import ToTensorV2


def preprocess_image(image: np.ndarray, width: int, height: int) -> torch.Tensor:
    """Препроцессинг имаджнетом.

    :param image: RGB изображение;
    :return: батч с одним изображением.
    """
    image = image.astype(np.float32)
    transforms = get_transforms(width=width, height=height)
    return transforms(image=image)['image']


def get_transforms(
    width: int,
    height: int,
) -> albu.Compose:
    """
    Returns a composition of image transformations for preprocessing and postprocessing.

    Args:
        width (int): The target width of the image.
        height (int): The target height of the image.
        text_size (int): The target size of the encoded text.
        preprocessing (bool): Flag indicating whether to apply preprocessing transformations.
        postprocessing (bool): Flag indicating whether to apply postprocessing transformations.

    Returns:
        TRANSFORM_TYPE: The composition of image transformations as an `albu.Compose` object.
    """

    return albu.Compose(
        [
            PadResizeOCR(
                target_height=height,
                target_width=width,
                mode='left',
            ),
            albu.Normalize(),
            ToTensorV2(),
        ],
    )


class PadResizeOCR:
    """
    Result in the desired size while maintaining the aspect ratio, adding padding if necessary
    """
    def __init__(self, target_width, target_height, dict_value: int = 0, mode: str = 'left'):
        self.target_width = target_width
        self.target_height = target_height
        self.dict_value = dict_value
        self.mode = mode

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:  # noqa: WPS210
        image = kwargs['image'].copy()

        h, w = image.shape[:2]

        tmp_w = min(int(w * (self.target_height / h)), self.target_width)
        image = cv2.resize(image, (tmp_w, self.target_height))

        dw = np.round(self.target_width - tmp_w).astype(int)
        if dw > 0:
            pad_left = 0
            pad_right = dw - pad_left
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        kwargs['image'] = image
        return kwargs
