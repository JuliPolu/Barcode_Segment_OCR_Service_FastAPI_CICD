import typing as tp
import numpy as np
import torch
from src.services.utils_ocr_prepostprocess import preprocess_image
from src.services.utils_ocr_predict import matrix_to_string


class BarcodeOCR:

    def __init__(self, config: tp.Dict):
        self._model_path = config['ocr_model_path']
        self._device = config['device']
        self._width: int = config['ocr_width']
        self._height: int = config['ocr_height']
        self._vocab: str = config['vocab']
        self._model: torch.nn.Module = torch.jit.load(self._model_path, map_location=self._device)

    def predict_barcode_text(self, image: np.ndarray) -> tp.List[str]:
        """Prediction of barcode bounding box

        Args:
        image: RGB image;

        Return:
            A dictionary containing the bounding box coordinates of the mask
        """

        return self._postprocess_predict(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Prediction of OCR model

        Args:
            image: RGB image

        Returns:
            ctc-matrix

        """
        image = preprocess_image(
            image,
            self._width,
            self._height,
        )
        return self._model(image[None].to(self._device)).cpu().detach()  # noqa: WPS221

    def _postprocess_predict(self, predict: torch.Tensor) -> np.ndarray:
        """Decode ctc-matrix to string

        Args:
            image: RGB image
            predict: output from model (ctc-matrix)

        Returns:
            Dict with predicted barcode numbers
        """

        string_pred, _ = matrix_to_string(predict, self._vocab)
        predicted_text = string_pred[0]

        return {'value': str(predicted_text)}
