from pathlib import Path


from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field
from typing import Dict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )


    """Paths Configuration"""

    BROADCAST_VIDEO_PATH :Path = Path("artifacts/broadcast.mp4")
    TACTICAM_VIDEO_PATH :Path = Path("artifacts/tacticam.mp4")
    FIELD_IMAGE : Path = Path("artifacts/soccer-green-field.jpg")
    OUTPUT_PATH : Path = Path("artifacts/unified_output.mp4")


    """Model Configuration"""
    PRETRAINED_YOLO_MODEL :Path = Path("artifacts/best.pt")
    TORCHREID_MODEL_NAME  :str = "osnet_x0_25"


    """Parameters"""
    FEATURE_WEIGHTS :Dict[str, float] = Field(default_factory=lambda: {
        "appearance": 0.3,
        "color_hist": 0.2,
        "field_coords": 0.4,
        "pose": 0.1,
    })





settings = Settings()