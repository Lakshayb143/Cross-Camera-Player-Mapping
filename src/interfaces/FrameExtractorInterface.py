from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Iterator

"""
abstract Class for implementing frame extraction strategies using design patterns.
"""

class FrameExtractingStrategy(ABC):
    """
    Interface for a frame extraction strategy.

    This defines the contract that all concrete extraction methods
    (like GPU or CPU-based) must follow.
    """

    @abstractmethod
    def __init__(self, video_path :Path):
        pass


    @abstractmethod
    def open_video_source(self) -> bool:
        """
        Opens the video source and returns True on success.
        """

    @abstractmethod
    def close_video_source(self) -> bool:
        """Releases the video source and any associated resources."""
         

    @abstractmethod
    def get_frames(self, frames_per_second :int) -> Iterator[np.ndarray]:
        """A generator that yields frames at a specified rate."""

    
    @property
    @abstractmethod
    def fps(self) -> float:
        """Returns the native frames per second of the video source."""

    
    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """Returns True if the video source is currently open."""
        

