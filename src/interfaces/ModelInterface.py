from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path


"""
abstract class for defining blue print of model loading.

"""

class ModelInterface(ABC):

    """
    Interface used to loading models.
    """
    @abstractmethod
    def __init__(self, model_path :Path, device : Optional[str] ):
        """
        Initializing the strategy with model loading with basic config.
        """
        pass

    