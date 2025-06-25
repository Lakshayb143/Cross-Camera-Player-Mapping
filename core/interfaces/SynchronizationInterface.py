from abc import ABC, abstractmethod
from typing import Tuple
from core.steps.FrameExtractor import FrameExtractor


"""
abstract class for implementing synchronization strategies using design patterns.
"""

class SynchronizationStrategy(ABC):
    """
    Interface for implementing any synchronization strategy.
    
    """

    @abstractmethod
    def find_offset(self, extractor_1 :FrameExtractor , extractor_2 :FrameExtractor) -> Tuple[int, float]:
        """
        It calculates temporal offset between two video streams.

        Returns:
            A tuple containing:
                a. offset_frames : int -> The number of frames video_2 is ahead of video_1.
                b. confidence : float -> A score indicating the confidence in the offset.

        """

        pass