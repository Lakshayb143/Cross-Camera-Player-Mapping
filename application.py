import cv2
import logging
import numpy as np
import cv2

from utils.logging import initialize_logging

from core.steps.FrameExtractor import FrameExtractor
from core.steps.Synchronizer import Synchronizer

from core.components.FrameExtractionStrategies import FfmpegcvCPUStrategy


from core.config import settings


"""Intializing logging """
initialize_logging()
logger = logging.getLogger(__name__)

def main():
    """Entry point of the application"""

    logger.info(f"\n\n\n\n")
    logger.info("-------------------------Starting application------------------------------------")



    logger.info("---Initializing Extractors ---")

    extractor_1 = FrameExtractor(settings.BROADCAST_VIDEO_PATH, FfmpegcvCPUStrategy)
    extractor_2 = FrameExtractor(settings.TACTICAM_VIDEO_PATH, FfmpegcvCPUStrategy)

    logger.info("--- Running Synchronization  ---")
    synchronizer = Synchronizer(extractor_1=extractor_1,extractor_2=extractor_2)

    try:
        synchronizer.sync()
        logger.info(
            f"Calculated Offset: {synchronizer.offset_frames} frames, Confidence: {synchronizer.confidence:.4f} "
        )
    except Exception as e:
        logger.error(f"Could not perform synchronization: {e}")
        return
    
    logger.info("--- Streaming Synchronized Frames ---")

    try:
        frame_count = 0
        for frame1, frame2 in synchronizer.get_synchronized_frames(fps=10):
            h1, w1, _ = frame1.shape
            h2, w2, _ = frame2.shape
            
            height = 540
            frame1_display = cv2.resize(frame1, (int(w1 * height / h1), height))
            frame2_display = cv2.resize(frame2, (int(w2 * height / h2), height))

            combined_frame = np.hstack((frame1_display, frame2_display))

            cv2.putText(combined_frame, 'Video 1 (Reference)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, 'Video 2 (Aligned)', (frame1_display.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Synchronized Videos', combined_frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested to quit.")
                break
        
        logger.info(f"Displayed {frame_count} synchronized frame pairs.")

    except Exception as e:
        logger.error(f"An error occurred during synchronized streaming: {e}")
    finally:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()