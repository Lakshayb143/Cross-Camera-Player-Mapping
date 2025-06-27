import cv2
import numpy as np
import logging

from src.config  import settings

logger = logging.getLogger(__name__)


"""List of coordinates of clicked point for homography"""
points = []

def mouse_callback(event, x, y, flags, params):
    """
    OpenCV mouse callback function.Appends (x,y) coordinates of a left click
    to global points list
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        logger.info(f"{len(points)} Points added with recent one = ({x},{y})")
        cv2.circle(params['frame'] ,(x,y) , 5, (0,255,0), -1)
        cv2.imshow("Points Selection for Homography", params['frame'])
        

def get_points():
    """
    Main function for loading video and displaying the first frame.
    So we can capture mouse clicks to select points for homography.
    """
    video_capture = cv2.VideoCapture(str(settings.TACTICAM_VIDEO_PATH))

    if not video_capture.isOpened():
        logger.error(f"Error could not open video file in points_utils.py with path : {settings.BROADCAST_VIDEO_PATH} ")
        return
    

    paused = False
    current_frame = None

    
    logger.info("\nInstructions:")
    logger.info("Please click on 4 points on the video frame in a specific order.")
    logger.info("For example, the 4 corners of the penalty box, starting top-left and moving clockwise.")
    logger.info("Press 'c' to confirm and print the points & 'q' to quit")
    
    cv2.namedWindow("Points Selection for Homography")
    

    while True:

        if not paused:
            ret, frame = video_capture.read()
            if not ret:
                logger.info("End of video reached....")
                break
            current_frame = frame.copy()
        
        display_frame = current_frame.copy()


        cv2.imshow("Points Selection for Homography", display_frame)
        cv2.setMouseCallback("Points Selection for Homography",mouse_callback, {'frame' : display_frame})
        key = cv2.waitKey(150 if not paused else 5000) & 0xFF

        if key == ord(' '):
            paused = not paused
            logger.info("Paused" if paused else "Resumed")

        if key == ord('q'):
            logger.info("you pressed q. leaving the window....")
            break

        if key == ord('c'):
            if len(points) < 4:
                logger.warning("You need to select at least 4 points")
            else:
                logger.info(f"\n\n Captured Source points : {points}")
                logger.info("Copy this line into your main script")
            break

    cv2.destroyAllWindows()
    video_capture.release()





    

