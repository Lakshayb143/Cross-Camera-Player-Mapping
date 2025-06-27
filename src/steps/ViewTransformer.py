import cv2
import logging
from typing import Tuple
import numpy as np



logger = logging.getLogger(__name__)

class ViewTransformer:
    """
    A class for handling normalization and homography.

    """

    def __init__(self, source_points : np.ndarray , destination_points : np.ndarray, target_resolution : Tuple):
        """
        Initializes the transformer by calculating the homography matrix.
        """
        if source_points.shape != destination_points.shape:
            logger.error("Source and destination points must have the same shape.")
            raise ValueError("Source and destination points must have the same shape.")
        
        logger.info("------ Calculating homography matrix =----------")
        threshold = 5.0
        self.homography_matrix, status = cv2.findHomography(source_points, destination_points, cv2.RANSAC, threshold)

        if self.homography_matrix is None:
            logger.error("Homography calculation with RANSAC failed. Check your source and destination points.")
            logger.info(f"source points : {source_points}")
            logger.info(f"destination points : {destination_points}")
            raise RuntimeError("Could not compute homography matrix.")
        
        self.target_resolution = target_resolution
        logger.info(f"Transformer initialized for target resolution {target_resolution}.")

        num_inliers = np.sum(status)
        logger.info(f"Homography calculated using RANSAC. Found {num_inliers} inliers out of {len(source_points)} points.")


    
    def transform(self , frame : np.ndarray) -> np.ndarray:
        """
        It applies the pre-calculated perspective warp and resizes the frame.
        """
        warped_frame = cv2.warpPerspective(frame, self.homography_matrix, self.target_resolution)

        return warped_frame


