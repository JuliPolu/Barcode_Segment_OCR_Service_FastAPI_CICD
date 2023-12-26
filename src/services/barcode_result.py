from typing import Dict

import numpy as np

from src.services.barcode_segmenter import BarcodeSegmenter
from src.services.barcode_ocr import BarcodeOCR
from src.services.crop_barcodes import crop_image


class BarcodeResult:

    def __init__(self, barcode_segmenter: BarcodeSegmenter, barcode_ocr: BarcodeOCR):
        self._barcode_segmenter = barcode_segmenter
        self._barcode_ocr = barcode_ocr

    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Prediction of barcode bbox coordinates and numbers

        Args:
        image: RGB image;

        Return:
            A dictionary containing barcode bbox coordinates and numbers
        """
        bbox_dict = self._barcode_segmenter.predict_barcode_bbox(image)
        if bbox_dict['bbox']['x_min']:
            cropped_image = crop_image(image, bbox_dict)
            ocr_result = self._barcode_ocr.predict_barcode_text(cropped_image)
        else:
            ocr_result = {'value': None}

        return {**bbox_dict, **ocr_result}
