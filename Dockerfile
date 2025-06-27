FROM python:3.11-slim

WORKDIR /app

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system-level dependencies.
# - ffmpeg: Required by ffmpegcv for video processing.
# - libgl1-mesa-glx, libglib2.0-0: Required by OpenCV for headless (no GUI) operation.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install uv
RUN uv pip install -r requirements.txt --system --no-cache-dir
# Copy the rest of the application source code into the container


# Create a directory for input data. We will mount our local data here.
# RUN mkdir -p /app/input_data

CMD ["uv", "--no-venv", "run", "application.py"]
