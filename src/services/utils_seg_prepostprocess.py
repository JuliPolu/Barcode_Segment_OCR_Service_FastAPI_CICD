import numpy as np
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import typing as tp
import cv2


def preprocess_image_segment(
    image: np.ndarray,
    target_image_size: int,
    encoder: str,
    pretrained: str = 'imagenet',
) -> torch.Tensor:
    """Preprecessing according to pretrained encoder

    Args:
        image: RGB-image
        target_image_size: target image size

    Returns:
        Processed tensor
    """
    image = image.astype(np.float32)
    processing_smp = smp.encoders.get_preprocessing_fn(encoder, pretrained=pretrained)
    preprocess = albu.Compose(
        [
            albu.Resize(height=target_image_size, width=target_image_size),
            albu.Lambda(image=processing_smp),
            ToTensorV2(),
        ],
    )
    image_array = preprocess(image=image)['image']

    return image_array.float().numpy()


def mask_to_bbox(mask: np.ndarray) -> dict:  # noqa: WPS210
    """
    Takes a binary mask as input and returns the bounding box coordinates of the mask.

    Args:
        mask: A numpy array representing the binary mask.

    Returns:
        A dictionary containing the bounding box coordinates of the mask.
    """

    if np.any(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_indices = np.where(rows)[0]
        y_min = row_indices[0]
        y_max = row_indices[-1]
        col_indices = np.where(cols)[0]
        x_min = col_indices[0]
        x_max = col_indices[-1]

        return {
            'bbox': {
                'x_min': int(x_min),
                'x_max': int(x_max),
                'y_min': int(y_min),
                'y_max': int(y_max),
            },
        }

    else:
        return {  # noqa: WPS503
            'bbox': {
                'x_min': None,
                'x_max': None,
                'y_min': None,
                'y_max': None,
            },
        }


def mask_postprocesing(
    prob_mask: np.ndarray,
    threshold: float,
    original_size: tp.Tuple[int, int, int],
) -> np.ndarray:
    """
    Applies image processing operations to a probability mask.

    Args:
        prob_mask: A numpy array representing the probability mask.
        threshold: A float value representing the threshold for converting the probability mask to a binary mask.
        original_size: A tuple of three integers representing the original size of the image.

    Returns:
       Processed mask. Only the largest connected component is set to 1 and all other components are set to
    """
    mask = create_binary_mask(prob_mask, threshold)
    mask = resize_mask(mask, original_size)
    return find_largest_connected_component(mask)


def create_binary_mask(prob_mask: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_mask > threshold).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: tp.Tuple[int, int]) -> np.ndarray:
    resized_mask = mask.transpose(1, 2, 0)
    return cv2.resize(resized_mask, size, interpolation=cv2.INTER_NEAREST)


def find_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels == 1:
        return labels_im.astype(np.uint8)
    else:
        max_label = find_label_of_largest_component(num_labels, labels_im)  # noqa: WPS503
        return (labels_im == max_label).astype(np.uint8)


def find_label_of_largest_component(num_labels: int, labels_im: np.ndarray) -> int:
    max_size = 0
    max_label = 0
    for label in range(1, num_labels):
        size = cv2.countNonZero((labels_im == label).astype(np.uint8))
        if size > max_size:
            max_size = size
            max_label = label

    return max_label
