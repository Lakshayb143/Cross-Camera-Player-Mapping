import logging
from typing import Iterator, Tuple, Type
import numpy as np

from src.interfaces.SynchronizationInterface import SynchronizationStrategy
from src.components.SynchronizationStrategies import CrossCorrelationSynchronizationStrategy
from src.steps.FrameExtractor import FrameExtractor

logger = logging.getLogger(__name__)

class Synchronizer:
    """
    Managing the synchronization of two video streams
    """
    def __init__(self , extractor_1 : FrameExtractor , extractor_2 :FrameExtractor, strategy : Type[SynchronizationStrategy] = CrossCorrelationSynchronizationStrategy):
        self.extractor_1 = extractor_1
        self.extractor_2 = extractor_2
        self.strategy = strategy()
        self.offset_frames = 0
        self.confidence = 0.0
        logger.info(f"Synchronizer initialized with strategy: {strategy.__name__}")

    def sync(self):
        """
        Calculates and stores the synchronization offset using the chosen strategy.
        """
        with self.extractor_1, self.extractor_2:
            self.offset_frames, self.confidence = self.strategy.find_offset(self.extractor_1, self.extractor_2)

    

    def get_synchronized_frames(self, fps :int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        A generator that yields synchronized frame pairs from both videos.
        Assumes video 1 is the reference.
        The offset determines how many frames to skip in video 2 at the start.

        Yields:
            a tuple (frame_from_video_1, frame_from_video_2)
        """

        if self.confidence < 0.5:
            logger.warning(
                f"Sync confidence is low ({self.confidence:.2f}). "
            )

        logger.info(f"Starting synchronized frame stream at {fps} FPS, with frame offset {self.offset_frames}.")

        with self.extractor_1, self.extractor_2:
            iterator_1 = self.extractor_1.extract(frames_per_second=fps)
            iterator_2 = self.extractor_2.extract(frames_per_second=fps)

            for _ in range(self.offset_frames):
                try:
                    next(iterator_2)
                except StopIteration:
                    logger.warning("Offset is larger than the number of frames in video 2. No frames to yield.")
                    return
                
            while True:
                try:
                    frame_1 = next(iterator_1)
                    frame_2 = next(iterator_2)

                    yield frame_1, frame_2
                except StopIteration:
                    logger.info("End of one or both video streams reached.")
                    break


        

            


