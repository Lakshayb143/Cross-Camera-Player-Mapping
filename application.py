import cv2
import logging
import numpy as np
import collections
import traceback
import os

from utils.logging_util import initialize_logging
from utils.points_utils import get_points

from src.steps.FrameExtractor import FrameExtractor
from src.steps.Synchronizer import Synchronizer
from src.steps.ViewTransformer import ViewTransformer
from src.steps.PlayerTracker import PlayerTracker
from src.steps.FeatureExtractor import FeatureExtractor
from src.steps.CrossViewMatcher import CrossViewMatcher

from src.IDManager import GlobalIdentityManager

from src.components.FrameExtractionStrategies import FfmpegcvCPUStrategy
from src.components.ModelStrategies import UltralyticsYoloModel, TorchReIDModel



from src.config import settings


"""Intializing logging """
initialize_logging()
logger = logging.getLogger(__name__)

def draw_unified_view(matched_pairs, unmatched1, unmatched2, id_manager, background_img):
    """Draws all players on a single top-down map with their global IDs."""
    if background_img is None or background_img.size == 0:
        raise ValueError("Background image is empty or None.")
    
    vis_img = background_img.copy()

    for p1, p2 in matched_pairs:
        gid = id_manager.register(p1, p2)
        coords = (p1['features']['field_coords'] + p2['features']['field_coords']) / 2
        coords = tuple(coords.astype(int))
        cv2.circle(vis_img, coords, 10, (0, 255, 0), -1)
        cv2.putText(vis_img, str(gid), (coords[0]-5, coords[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for p in unmatched1 + unmatched2:
        gid = id_manager.get_global_id(p['view'], p['track_id'])
        coords = p['features']['field_coords']
        cv2.circle(vis_img, tuple(coords.astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(vis_img, str(gid), tuple((coords + np.array([-5, -15])).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return vis_img

    return vis_img
def main():
    """Entry point of the application"""

    logger.info(f"\n\n\n\n")
    logger.info("-------------------------Starting application------------------------------------")


    """
    Defining Homography points
    These points were acquired using 'points_utils.py' for both videos.
    """
    logger.info("Defining Homography points using points_utils.py' for both videos.")
    broadcast_points = np.float32([
    [135, 555], [375, 572], [403, 491], [581, 505],
    [404, 491], [658, 427], [630, 506], [774, 433],
    [838, 739], [950, 696], [1069, 654], [1179, 617],
    [62, 709], [340, 731], [857, 591], [626, 574],
    [247, 659], [473, 606], [245, 521], [469, 478],
    [398, 677], [297, 669], [531, 615], [623, 623]
    ])

    tacticam_points = np.float32([
    [110, 232], [230, 244], [230, 144], [322, 149],
    [287, 512], [343, 451], [381, 410], [424, 362],
    [142, 312], [276, 324], [255, 233], [371, 239],
    [41, 413], [31, 349], [129, 285], [137, 343],
    [6, 469], [165, 486], [242, 299], [374, 313],
    [24, 407], [82, 413], [124, 339], [178, 344]
    ])



    width, height = 1920, 1080
    destination_points = np.float32([
    [94, 237], [93, 63], [306, 237], [304, 63],
    [307, 577], [307, 510], [307, 446], [307, 397],
    [94, 596], [163, 596], [94, 382], [163, 382],
    [94, 735], [306, 735], [94, 238], [306, 238],
    [94, 382], [164, 382], [94, 238], [164, 238],
    [94, 595], [165, 595], [94, 735], [165, 735]
    ])
    

    logger.info("Initializing all modules")
    transformer_1 :ViewTransformer = ViewTransformer(broadcast_points, tacticam_points, (width, height))
    player_tracker = PlayerTracker(UltralyticsYoloModel)
    feature_extractor = FeatureExtractor(TorchReIDModel,UltralyticsYoloModel)

    logger.info("---Initializing Extractors ---")

    extractor_1 = FrameExtractor(settings.BROADCAST_VIDEO_PATH, FfmpegcvCPUStrategy)
    extractor_2 = FrameExtractor(settings.TACTICAM_VIDEO_PATH, FfmpegcvCPUStrategy)

    logger.info("--- Running Synchronization  ---")
    synchronizer = Synchronizer(extractor_1=extractor_1,extractor_2=extractor_2)

    matcher = CrossViewMatcher(settings.FEATURE_WEIGHTS, max_cost_threshold=0.75)
    id_manager = GlobalIdentityManager()
    

    try:
        synchronizer.sync()
        field_map = cv2.imread("field.jpg")
        field_map = cv2.resize(field_map, (width, height))
        logger.info(
            f"Calculated Offset: {synchronizer.offset_frames} frames, Confidence: {synchronizer.confidence:.4f} "
        )
    except Exception as e:
        logger.error(f"Could not perform synchronization: {e}")
        logger.error(traceback.format_exc())
        return
    
    logger.info("--- PHASE 1 : Extracting data from all frames ---")
    try:
        all_player_data_by_frame = collections.defaultdict(list)
        frame_index = 0
        for frame1, frame2 in synchronizer.get_synchronized_frames(fps=10):

            tracked_players_v1 = player_tracker.track_players(frame1)
            for box, track_id, conf in tracked_players_v1:
                features = feature_extractor.extract_features(frame1,box,transformer_1)
                all_player_data_by_frame[frame_index].append({"view": "broadcast", "track_id": track_id, "features": features})   

            tracked_players_v2 = player_tracker.track_players(frame2)
            for box, track_id, conf in tracked_players_v2:
                features = feature_extractor.extract_features(frame2,box,transformer_1)
                all_player_data_by_frame[frame_index].append({"view": "tacticam", "track_id": track_id, "features": features})

            frame_index += 1

        logger.info("--- PHASE 1: Data extraction Completed.")

        logger.info("\n\nPHASE 2: Matching players and visualizing results...")
        result_saver = cv2.VideoWriter_fourcc(*'mp4v')
        result_video = cv2.VideoWriter(str(settings.OUTPUT_PATH), result_saver, 10, (field_map.shape[1], field_map.shape[0]))
        for i in range(frame_index):
            frame_data = all_player_data_by_frame[i]
            players_1 = [p for p in frame_data if p["view"] == "broadcast"]
            players_2 = [p for p in frame_data if p["view"] == "tacticam"]

            matched, unmatched1, unmatched2 = matcher.match_players_in_frame(players_1, players_2)

            vis_frame = draw_unified_view(matched, unmatched1, unmatched2, id_manager, field_map)
            cv2.imshow("Unified Top-Down View", vis_frame)
            result_video.write(vis_frame)
            if cv2.waitKey(1000) & 0xFF == ord('q'): 
                break
            
        logger.info("Execution Completed.")
    except Exception as e:
        logger.error(f"An error occurred during synchronized streaming: {e}")
        logger.error(traceback.format_exc())
    finally:
        result_video.release()
        logger.info(f"Unified visualization saved to: {settings.OUTPUT_PATH}")
        cv2.destroyAllWindows()    



if __name__ == "__main__":
    main()