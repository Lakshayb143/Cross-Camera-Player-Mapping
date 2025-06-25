import logging
from typing import Tuple, List
import numpy as np
from skimage.metrics import structural_similarity
import cv2

from core.interfaces.SynchronizationInterface import SynchronizationStrategy
from core.steps.FrameExtractor import FrameExtractor

logger = logging.getLogger(__name__)

"""
Here we will implement synchronization strategies.
"""

class CrossCorrelationSynchronizationStrategy(SynchronizationStrategy):
    """
    Find the offset by calculating the Structural Similarity Index between frame sequences.
    """

    def __init__(self, sample_duration_sec :int = 5, search_window_sec : int = 0.5 , fps :int = 15):
        self.sample_duration_sec = sample_duration_sec
        self.search_window_sec = search_window_sec
        self.fps = fps
        logger.info(
            f"Initialized Cross Correlation Synchronization Strategy with sample_duration = {sample_duration_sec}s , search_window = {search_window_sec}s, comparison_fps = {fps}",
        )
    

    def extract_and_preprocess_frame(self, extractor : FrameExtractor , duration :int) -> List[np.ndarray]:
        """
        Extracts, resizes, and converts frames to grayscale for faster comparison
        """

        frames = []
        count = 0
        target_frame_count = duration * self.fps

        for frame in extractor.extract(frames_per_second=self.fps):
            if count >= target_frame_count:
                break

            gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            resized_frame =  cv2.resize(gray_frame, (320,180))
            frames.append(resized_frame)
            count += 1

        return frames
    
    def find_offset(self, extractor_1: FrameExtractor, extractor_2: FrameExtractor) -> Tuple[int, float]:
        logger.info("Starting visual cross-correlation to find offset.")

        logger.info(f"Extracting {self.sample_duration_sec}s sample from first video.")

        ref_frames = self.extract_and_preprocess_frame(extractor=extractor_1, duration=self.sample_duration_sec)
        if not ref_frames:
            logger.error("Could not extract any frames from the reference video.")
            return (0,0.0)
        
        logger.info(f"Extracting {self.search_window_sec}s search window from second video.")
        
        search_frames = self.extract_and_preprocess_frame(extractor=extractor_2, duration=self.search_window_sec)
        if not search_frames:
            logger.error("Could not extract any frames from the search video.")
            return (0,0.0)
        
        num_ref_frames = len(ref_frames)
        num_search_frames = len(search_frames)

        if num_search_frames < num_ref_frames:
            logger.error("Search window is smaller than reference sample. Cannot find offset.")
            return (0,0.0)
        
        mean_ssim_scores = []
        max_possible_offset = num_search_frames - num_ref_frames
        logger.info(f"Comparing sequences. Max frame offset to check: {max_possible_offset}")

        for offset in range(max_possible_offset):
            current_ssim_sum = 0
            for i in range(num_ref_frames):
                score, _ = structural_similarity(ref_frames[i], search_frames[i + offset], full = True)
                current_ssim_sum += score
            
            mean_ssim_scores.append(current_ssim_sum / num_ref_frames)

        
        if not mean_ssim_scores:
            logger.warning("Could not calculate any ssi score")
            return (0,0.0)
        
        best_match_offset_frame = np.argmax(mean_ssim_scores)
        best_confidence = np.max(mean_ssim_scores)

        logger.info(
            f"Synchronization complete. Best match found at frame offset: {best_match_offset_frame} & with confidence: {best_confidence:.4f} "
        )

        return int(best_match_offset_frame), float(best_confidence)


        
