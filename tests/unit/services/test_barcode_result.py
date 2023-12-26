import random
from copy import deepcopy

import cv2
import numpy as np

from src.containers.containers import AppContainer


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


class FakeBarcodeOCR:

    def predict_barcode_text(self, image):
        return  {'value': '5053990125050'}


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    with app_container.reset_singletons():
        with app_container.barcode_segmenter.override(FakeBarcodeSegmenter()), \
                app_container.barcode_ocr.override(FakeBarcodeOCR()):
            barcode_result = app_container.barcode_result()
            barcode_result.predict(sample_image_np)


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    with app_container.reset_singletons():
        with app_container.barcode_segmenter.override(FakeBarcodeSegmenter()), \
                app_container.barcode_ocr.override(FakeBarcodeOCR()):
            initial_image = deepcopy(sample_image_np)
            barcode_result = app_container.barcode_result()
            barcode_result.predict(sample_image_np)

            assert np.allclose(initial_image, sample_image_np)