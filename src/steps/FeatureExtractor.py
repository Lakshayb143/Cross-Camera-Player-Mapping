import cv2
import logging
import torchreid as TRE
import numpy as np
import torch
from PIL import Image

from src.config import settings
from src.interfaces.ModelInterface import ModelInterface
from src.components.ModelStrategies import UltralyticsYoloModel
from src.components.ModelStrategies import TorchReIDModel
from src.steps.ViewTransformer import ViewTransformer

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    It is for extracting rich feature vector for each player detected.
    """


    def __init__(self, reid_model_loader : ModelInterface , model_loader : ModelInterface ):
        """
        Initializes the feature extractor and loads necessary models.
        """
        self.reid_model_loader = reid_model_loader()
        self.model_loader = model_loader()
        logger.info("Feature Extractor Initialized succesfully.")


    def extract_appearance_embedding(self, frame : np.ndarray, box : np.ndarray) -> np.ndarray:
        """
        Extracts a deep learning-based Re-ID feature vector.
        """
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            logger.info("Crop is zero. appearance embedding is zero.")
            return np.zeros(512) 
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        
        img = self.reid_model_loader.ried_transfrom(pil_crop).unsqueeze(0).to(self.reid_model_loader.device)
        with torch.no_grad():
            embedding = self.reid_model_loader.reid_model(img)

        return embedding.cpu().numpy().flatten()
    
    def extract_color_histogram(self, frame : np.ndarray, box : np.ndarray) -> np.ndarray:
        """
        Computes a color histogram for the torso region
        """
        x1, y1, x2, y2 = map(int, box)

        torso_y1 = y1 + int((y2 - y1) * 0.2)
        torso_y2 = y1 + int((y2 - y1) * 0.6)
        torso_crop = frame[torso_y1:torso_y2, x1:x2]

        if torso_crop.size == 0:
            return np.zeros(16 + 16 + 16) 
        
        hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for H, S, V channels
        h_hist = cv2.calcHist([hsv_crop], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv_crop], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv_crop], [2], None, [16], [0, 256])
        
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        cv2.normalize(v_hist, v_hist)
        
        return np.concatenate((h_hist, s_hist, v_hist)).flatten()
    


    def get_field_coordinates(self, box :np.ndarray, transformer :ViewTransformer) -> np.ndarray:
        box_center_x = (box[0] + box[2]) / 2
        box_bottom_y = box[3]
        player_pos_pixel = np.float32([[[box_center_x, box_bottom_y]]])
        
        warped_pos = cv2.perspectiveTransform(player_pos_pixel, transformer.homography_matrix)
        return warped_pos.flatten()
    

    def extract_pose_keypoints(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        results = self.model_loader.model(frame)
        if results[0].keypoints and results[0].keypoints.xy.shape[1] > 0:
            return results[0].keypoints.xyn.cpu().numpy().flatten()
        
        return np.zeros(17 * 2)
    

    def extract_features(self, frame: np.ndarray, box: np.ndarray, transformer: ViewTransformer) -> dict:
        """
        Runs all feature extractors for a given player box.
        """

        feature_embedding = {
            "appearance" : self.extract_appearance_embedding(frame,box),
            "color_hist" : self.extract_color_histogram(frame,box),
            "field_coords" : self.get_field_coordinates(box, transformer),
            "pose" : self.extract_pose_keypoints(frame,box)
        }

        return feature_embedding

