import numpy as np
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import typing as tp
import cv2


def preprocess_image_segment(
    image: np.ndarray,
    target_image_size: int,
) -> torch.Tensor:
    """Preprecessing according to pretrained encoder

    Args:
        image: RGB-image
        target_image_size: target image size

    Returns:
        Processed tensor
    """
    image = image.astype(np.float32)
    preprocess = albu.Compose(
        [
            albu.Resize(height=target_image_size, width=target_image_size),
            albu.Normalize(),
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

    bbox_list = []
    num_labels = mask.max()

    for label in range(1, num_labels + 1):  # Start from 1 as 0 is the background
        label_mask = (mask == label).astype(np.uint8)

        if np.any(label_mask):
            rows = np.any(label_mask, axis=1)
            cols = np.any(label_mask, axis=0)
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]

            if row_indices.size > 0 and col_indices.size > 0:
                y_min, y_max = row_indices[0], row_indices[-1]
                x_min, x_max = col_indices[0], col_indices[-1]

                bbox_list.append({
                    'bbox': {
                        'x_min': int(x_min),
                        'x_max': int(x_max),
                        'y_min': int(y_min),
                        'y_max': int(y_max),
                    },
                })
        else:
            bbox_list.append({'bbox': None})

    return bbox_list


def mask_postprocesing(
    prob_mask: np.ndarray,
    threshold_prob: float,
    original_size: tp.Tuple[int, int, int],
    threshold_size: int,
) -> np.ndarray:
    """
    Applies image processing operations to a probability mask.

    Args:
        prob_mask: A numpy array representing the probability mask.
        threshold_prob: A float value representing the threshold for converting the probability mask to a binary mask.
        original_size: A tuple of three integers representing the original size of the image.

    Returns:
       Processed mask. Only the largest connected component is set to 1 and all other components are set to
    """
    mask = create_binary_mask(prob_mask, threshold_prob)
    mask = resize_mask(mask, original_size)
    return find_barcodes(mask, threshold_size)


def create_binary_mask(prob_mask: np.ndarray, threshold_prob: float) -> np.ndarray:
    return (prob_mask > threshold_prob).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: tp.Tuple[int, int]) -> np.ndarray:
    resized_mask = mask.transpose(1, 2, 0)
    return cv2.resize(resized_mask, size, interpolation=cv2.INTER_NEAREST)


def find_barcodes(mask: np.ndarray, threshold_size: int) -> np.ndarray:
    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels == 1:
        return labels_im.astype(np.uint8)
    else:
        return find_labels_psssing_threshold(threshold_size, num_labels, labels_im)  # noqa: WPS503


def find_labels_psssing_threshold(threshold_size: int, num_labels: int, labels_im: np.ndarray) -> tp.List[int]:
    output_mask = np.zeros_like(labels_im, dtype=np.uint8)
    for label in range(1, num_labels):
        size = cv2.countNonZero((labels_im == label).astype(np.uint8))
        if size > threshold_size:
            output_mask[labels_im == label] = label
    return output_mask
