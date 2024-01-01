from omegaconf import OmegaConf
from pydantic import BaseModel


class SegmenterConfig(BaseModel):
    model_path: str
    device: str
    ort_provider: str
    img_input_size: int
    threshold_prob: float
    threshold_size: int


class OCRConfig(BaseModel):
    model_path: str
    device: str
    width: int
    height: int
    vocab: str


class Config(BaseModel):
    segmenter: SegmenterConfig
    ocr: OCRConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
