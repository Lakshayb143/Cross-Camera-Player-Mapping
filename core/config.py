from pathlib import Path


from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )


    """Paths Configuration"""

    BROADCAST_VIDEO_PATH :Path = Path("artifacts/broadcast.mp4")
    TACTICAM_VIDEO_PATH :Path = Path("artifacts/tacticam.mp4")


    """Model Configuration"""
    PRETRAINED_YOLO_MODEL :Path = Path("artifacts/best.pt")





settings = Settings()