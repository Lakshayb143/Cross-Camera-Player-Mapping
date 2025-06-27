import cv2
import logging
import numpy as np
from typing import List

from src.config import settings

from src.interfaces.ModelInterface import ModelInterface
from src.components.ModelStrategies import UltralyticsYoloModel

logger = logging.getLogger(__name__)

class PlayerTracker:
    """
    It handles player detection and tracking within a single video view.
    """

    def __init__(self, model_loader : ModelInterface):
        """
        Initializing tracker with yolo model
        """
        self.model_loader : ModelInterface = model_loader()
        

    
    def track_players(self, frame : np.ndarray, confidence_threshold :float = 0.4) -> List:
        """
        Performs detection and tracking in a single view.
        """

        results = self.model_loader.model.track(frame)

        tracked_players = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, confidence, class_id in zip(boxes, track_ids, confidences, class_ids):
                if confidence >= confidence_threshold:
                    tracked_players.append((box, track_id, confidence))
        
        return tracked_players
    

    @staticmethod
    def draw_tracks(frame: np.ndarray, tracked_players: list):
        """
        A helper function to visualize the tracks on a frame.
        """
        for box, track_id, conf in tracked_players:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Player_ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
