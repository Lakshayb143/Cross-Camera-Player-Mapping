# import torchreid
# torchreid.models.show_avai_models()

from src.config import settings

import cv2

cap = cv2.VideoCapture(str(settings.TACTICAM_VIDEO_PATH))

success, tacticam_frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Failed to read frame from tacticam video")

height, width = tacticam_frame.shape[:2]
target_resolution = (width, height)

print(f"{width}=")
print(f"{height}=")
