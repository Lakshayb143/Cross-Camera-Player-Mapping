import cv2
import logging

from utils.logging import initialize_logging

from core.steps.FrameExtractor import FrameExtractor
from core.components.FrameExtractionStrategies import FfmpegcvCPUStrategy
from core.config import settings


"""Intializing logging """
initialize_logging()
logger = logging.getLogger(__name__)

def main():
    """Entry point of the application"""

    logger.info("-------------------------Starting application------------------------------------")
    logger.info("Starting frame extraction..")

    try:
        with FrameExtractor(settings.BROADCAST_VIDEO_PATH, strategy=FfmpegcvCPUStrategy) as extractor:
            frame_count = 0

            for frame in extractor.extract(frames_per_second=5):
                frame_count += 1
                # print(frame)
                logger.info(f"Successfully extracted {frame_count} frames")
    
    except Exception as e:
        logger.error(f"Error occurred during the extraction process: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()