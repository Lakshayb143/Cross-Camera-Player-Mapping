import torch
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import torchreid as TRE

from ultralytics import YOLO

from src.interfaces.ModelInterface import ModelInterface

from src.config import settings
logger = logging.getLogger(__name__)

class UltralyticsYoloModel(ModelInterface):
    """
    It will load the provided pretrained yolo v11 model

    """

    def __init__(self, model_path: Path = settings.PRETRAINED_YOLO_MODEL, device: str | None = "cpu"):
        self.model_path = model_path
        self.device = device
        try:
            self.model = YOLO(str(self.model_path))
            logger.info(f"Model loaded successfully with path :{model_path} & device : {device} with strategy: {self.__class__.__name__}")
        except Exception as e:
            logger.critical(f"Error in loading model with path :{model_path} : {e}")


class TorchReIDModel(ModelInterface):
    """
    It will load the torchreid model.
    """

    def __init__(self, reid_model_name :str = settings.TORCHREID_MODEL_NAME, device :str | None = "cpu"):
        logger.info(f"Loading Re-Id Model :{reid_model_name}")
        self.reid_model = TRE.models.build_model(
            name=reid_model_name,
            num_classes=1,
            pretrained=True
        )
        self.reid_model.eval()
        self.device = device
        self.reid_model.to(self.device)
        _, self.ried_transfrom = TRE.data.transforms.build_transforms(is_train=False, height=256, width=128)
        logger.info(f"Re-Id Model loaded successfully with model : {reid_model_name}")

