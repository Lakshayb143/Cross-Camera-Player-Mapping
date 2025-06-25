import torch
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict

from ultralytics import YOLO

from core.interfaces.ModelInterface import ModelInterface

from core.config import settings
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

    
    def detect(self , frame : np.ndarray) -> List[Dict]:
        return self.model.predict(frame)



