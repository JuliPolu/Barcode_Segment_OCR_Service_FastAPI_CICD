from typing import List
from fastapi import Depends, File, UploadFile
from dependency_injector.wiring import Provide, inject
from src.containers.containers import AppContainer
from src.routes.routers import router
from src.services.barcode_result import BarcodeResult
import cv2
import numpy as np


@router.post('/predict')
@inject
def predict(
    images: List[UploadFile] = File(...),  # noqa: B008, WPS404
    service: BarcodeResult = Depends(Provide[AppContainer.barcode_result]),  # noqa: B008, WPS404
):
    final_results = {}

    for image_file in images:
        image_contents = image_file.file.read()
        img = cv2.imdecode(np.frombuffer(image_contents, np.uint8), cv2.IMREAD_COLOR)

        final_result = service.predict(img)
        final_results[image_file.filename] = final_result

    return final_results


@router.get('/health_check')
def health_check():
    return 'OK'
