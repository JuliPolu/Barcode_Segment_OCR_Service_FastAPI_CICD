from src.config import SegmenterConfig
import typing as tp

import numpy as np
import torch
import onnxruntime as ort

from src.services.utils_seg_prepostprocess import preprocess_image_segment, mask_postprocesing, mask_to_bbox


class BarcodeSegmenter:

    def __init__(self, config: SegmenterConfig):
        self._config = config
        self._ort_session = ort.InferenceSession(
            self._config['model_path'],
            providers=[self._config['ort_provider']],
        )

    def predict_barcode_bbox(self, image: np.ndarray) -> tp.List[str]:
        """Prediction of barcode bounding box

        Args:
        image: RGB image;

        Return:
            A dictionary containing the bounding box coordinates of the mask
        """
        return self._postprocess_predict(image, self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Prediction of Prababilistic mask of size that outputs model

        Args:
            image: RGB image

        Returns:
            Mask with probabilities of barcode location
        """
        image = preprocess_image_segment(
            image,
            self._config['img_input_size'],
        )
        input_name = self._ort_session.get_inputs()[0].name
        output = self._ort_session.run(None, {input_name: image[None]})
        prob_mask = torch.sigmoid(torch.tensor(output[0]))[0]
        return prob_mask.numpy()

    def _postprocess_predict(self, image: np.ndarray, predict: np.ndarray) -> np.ndarray:
        """Postprocessing for getting binary segmentation mask of original size

        Args:
            image: RGB image
            predict: output from model (ask with probabilities of barcode location)

        Returns:
            Binary mask of barcode location
        """
        original_size = (image.shape[1], image.shape[0])
        mask = mask_postprocesing(
            predict,
            self._config['threshold_prob'],
            original_size,
            self._config['threshold_size'],
        )

        return mask_to_bbox(mask)
