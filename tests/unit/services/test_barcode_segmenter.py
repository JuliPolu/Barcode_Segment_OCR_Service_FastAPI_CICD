import random
from copy import deepcopy

import numpy as np

from src.containers.containers import AppContainer


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    barcode_segmenter = app_container.barcode_segmenter()
    barcode_segmenter.predict_barcode_bbox(sample_image_np)


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    initial_image = deepcopy(sample_image_np)
    barcode_segmenter = app_container.barcode_segmenter()
    barcode_segmenter.predict_barcode_bbox(sample_image_np)

    assert np.allclose(initial_image, sample_image_np)


def test_predict_handles_empty_images(app_container: AppContainer):
    barcode_segmenter = app_container.barcode_segmenter()
    barcode = barcode_segmenter.predict_barcode_bbox(np.random.randint(0, 256, (550, 550, 3), dtype=np.uint8))

    assert barcode['bbox']['x_min'] == None
