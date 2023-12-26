import os.path
from typing import List
import cv2
import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from io import BytesIO
from app import set_routers
from src.containers.containers import AppContainer
from src.routes import barcode as barcode_routes

TESTS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def sample_image_bytes():
    with open(os.path.join(TESTS_DIR, 'images', '77.jpg'), 'rb') as f:   # noqa: WPS515
        return f.read()


@pytest.fixture(scope='session')
def sample_images_bytes() -> List[UploadFile]:
    upload_files = []

    image_paths = [os.path.join(TESTS_DIR, 'images', '77.jpg'), os.path.join(TESTS_DIR, 'images', '777.jpg')]

    for image_path in image_paths:
        with open(image_path, 'rb') as image_file:
            file_content = BytesIO(image_file.read())
            upload_file = UploadFile(filename=os.path.basename(image_path),
                                     content_type='image/jpeg',
                                     file=file_content)
            upload_files.append(upload_file)

    return upload_files


@pytest.fixture
def sample_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, 'images', '77.jpg'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@pytest.fixture
def sample_crop_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, 'images', '77_crop.jpg'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope='session')
def app_config():
    return OmegaConf.load(os.path.join(TESTS_DIR, 'test_config.yml'))


@pytest.fixture
def app_container(app_config):
    container = AppContainer()
    container.config.from_dict(app_config)
    return container


@pytest.fixture
def wired_app_container(app_config):
    container = AppContainer()
    container.config.from_dict(app_config)
    container.wire([barcode_routes])
    yield container
    container.unwire()


@pytest.fixture
def test_app(app_config, wired_app_container):
    app = FastAPI()
    set_routers(app)
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
