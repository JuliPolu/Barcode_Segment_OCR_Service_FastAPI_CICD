import random
from copy import deepcopy

import cv2
import numpy as np

from src.containers.containers import AppContainer
from src.services.crop_barcodes import crop_image


class FakeBarcodeSegmenter:

    def predict_barcode_bbox(self, image):
        return  {
            'bbox': {
                'x_min': 203,
                'x_max': 717,
                'y_min': 572,
                'y_max': 939,
            },
        }


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    with app_container.reset_singletons():
        with app_container.barcode_segmenter.override(FakeBarcodeSegmenter()):
            barcode_segmenter = app_container.barcode_segmenter()
            bbox_dict = barcode_segmenter.predict_barcode_bbox(sample_image_np)
            cropped_image = crop_image(sample_image_np, bbox_dict)
            barcode_ocr = app_container.barcode_ocr()
            barcode_ocr.predict_barcode_text(cropped_image)


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_crop_image_np: np.ndarray):
    initial_image = deepcopy(sample_crop_image_np)
    barcode_ocr = app_container.barcode_ocr()
    barcode_ocr.predict_barcode_text(sample_crop_image_np)

    assert np.allclose(initial_image, sample_crop_image_np)


def test_predict_text_length(app_container: AppContainer, sample_crop_image_np: np.ndarray):
    barcode_ocr = app_container.barcode_ocr()
    ocr_result = barcode_ocr.predict_barcode_text(sample_crop_image_np)

    assert len(ocr_result['value']) == 13
