from fastapi.testclient import TestClient
from http import HTTPStatus
from io import BytesIO
import base64


def test_classes_list(client: TestClient):
    response = client.get('/barcode/health_check')
    assert response.status_code == HTTPStatus.OK


def test_predict(client: TestClient, sample_images_bytes: bytes):
    files = [('images', (image.filename, image.file.read(), image.content_type)) for image in sample_images_bytes]
    
    response = client.post('/barcode/predict', files=files)

    assert response.status_code == HTTPStatus.OK

    predicted_value = response.json()['77.jpg']['value']

    assert isinstance(predicted_value, str)
    assert len(predicted_value)==13

