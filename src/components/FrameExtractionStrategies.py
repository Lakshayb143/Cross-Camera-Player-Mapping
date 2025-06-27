from typing import Iterator, Optional
import ffmpegcv
import numpy as np
from pathlib import Path
import logging

from src.interfaces.FrameExtractorInterface import FrameExtractingStrategy

logger = logging.getLogger(__name__)

"""
We will implement Stratagies for frame extraction.
"""


class FfmpegcvCPUStrategy(FrameExtractingStrategy):
    """
    Concrete frame extraction strategy using ffmpegcv with CPU decoding.
    """

    def __init__(self, video_path : Path):
        self.video_path :Path = video_path
        self._video_capture : Optional[ffmpegcv.VideoCapture] = None
    

    def open_video_source(self) -> bool:
        try:
            logger.info(f"Attempting to open '{self.video_path}' with CPU.")
            self._video_capture = ffmpegcv.VideoCapture(str(self.video_path))
            logger.info(f"Video is opened on '{self.video_path}' with CPU.")
            return self._video_capture.isOpened()
        except Exception as e:
            print(f"Error opening video with CPU: {e}")
            self._video_capture = None
            return False
        
    
    def close_video_source(self) -> bool:
        if self._video_capture:
            logger.debug("Releasing video capture source.")
            self._video_capture.release()
            self._video_capture = None
    

    def get_frames(self, frames_per_second: int) -> Iterator[np.ndarray]:
        if not self.is_opened:
            logger.error("Video source is not open. Call open_video_source() first.")
            raise RuntimeError("Video source is not open. Call open_video_source() first.")
        
        source_fps = self.fps
        if source_fps == 0:
            logger.error("The FPS of source video couldn't be determined")
            raise ValueError("The FPS of source video couldn't be determined")
        
        frame_interval = int(source_fps/ frames_per_second)
        frame_counter = 0

        while self.is_opened:
            ret, frame = self._video_capture.read()

            if not ret:
                logger.info("End of video stream reached.")
                break

            if frame_counter % frame_interval == 0:
                logger.debug(f"Yeilding frame number {frame_counter}")
                yield frame
            
            frame_counter += 1

            

    @property
    def fps(self) -> float:
        return self._video_capture.fps if self._video_capture else 0.0


    @property
    def is_opened(self) -> bool:
        return self._video_capture is not None and self._video_capture.isOpened()



