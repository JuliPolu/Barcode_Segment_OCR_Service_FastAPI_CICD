from dependency_injector import containers, providers

from src.services.barcode_segmenter import BarcodeSegmenter
from src.services.barcode_ocr import BarcodeOCR
from src.services.barcode_result import BarcodeResult


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    barcode_segmenter = providers.Singleton(
        BarcodeSegmenter,
        config=config.services.barcode_segmenter,
    )

    barcode_ocr = providers.Singleton(
        BarcodeOCR,
        config=config.services.barcode_ocr,
    )

    barcode_result = providers.Singleton(
        BarcodeResult,
        barcode_segmenter=barcode_segmenter,
        barcode_ocr=barcode_ocr,
    )
