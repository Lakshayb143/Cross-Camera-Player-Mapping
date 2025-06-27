import logging
from pathlib import Path
import numpy as np
from typing import Iterator,Type


from src.components.FrameExtractionStrategies import FfmpegcvCPUStrategy
from src.interfaces.FrameExtractorInterface import FrameExtractingStrategy

logger = logging.getLogger(__name__)

class FrameExtractor:
    """
    A class for implementing fram extraction
    """

    def __init__(self, video_path : Path, strategy : Type[FrameExtractingStrategy]):
        logger.info(f"Initializing extractor for {video_path}  with primary strategy {strategy.__name__}")
        self.strategy = strategy(video_path=video_path)
    

    def __enter__(self):

        if not self.strategy.open_video_source():
            logger.error(f"{self.strategy.__class__.__name__} failed to open video {self.strategy.video_path}")
            raise IOError(f"Failed to open video with path : {self.strategy.video_path}")
    
        logger.info(f"Successfully opened video with active strategy: {self.strategy.__class__.__name__}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Closing Video source")
        self.strategy.close_video_source()

        if exc_type:
            logger.error(f"Exception during frame extraction: {exc_val}")
            logger.error(f"Traceback to the exception : {exc_tb}")

    
    def extract(self, frames_per_second : int = 15) -> Iterator[np.ndarray]:
        if not self.strategy.is_opened:
            logger.error("Video source is not open.")
            raise RuntimeError("Video source is not open.")
        

        logger.info(f"Starting frame extraction at {frames_per_second} FPS.")
        yield from self.strategy.get_frames(frames_per_second)
