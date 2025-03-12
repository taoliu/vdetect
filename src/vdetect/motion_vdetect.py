import cv2
import numpy as np
import time
import argparse
import os
from collections import deque

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Motion detection script for ESP32-CAM")
parser.add_argument("--url", type=str, required=True, help="ESP32-CAM stream URL")
parser.add_argument("--name", type=str, default="camera", help="Camera name for filename prefix. Default: camera")
parser.add_argument("--buffer", type=int, default=15, help="Seconds of video to store before motion. Default: 15")
parser.add_argument("--post_motion", type=int, default=15, help="Seconds to record after motion stops. Default: 15")
parser.add_argument("--max_record", type=int, default=45, help="Maximum recording time in seconds. Default: 45")
parser.add_argument("--motion_threshold", type=float, default=0.1, help="Motion detection sensitivity (0-1). Default:0.1")
parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate video clockwise (0, 90, 180, 270). Default: 0")
parser.add_argument("--save_dir", type=str, default=".", help="Directory to save recorded videos. Default: . (which means the current directory)")

args = parser.parse_args()

# Assign variables from arguments
CAMERA_URL = args.url
CAMERA_NAME = args.name
BUFFER_SECONDS = args.buffer
POST_MOTION_SECONDS = args.post_motion
MAX_RECORD_SECONDS = args.max_record
MIN_MOTION_RATIO = args.motion_threshold
ROTATION_ANGLE = args.rotate
SAVE_DIRECTORY = args.save_dir

# Ensure the save directory exists
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

# Frame storage for buffering
frame_buffer = deque(maxlen=BUFFER_SECONDS * 10)  # Placeholder, will update FPS dynamically
recording_frames = []
reference_frame = None
motion_detected = False
recording = False
record_start_time = None  # Track when recording starts
last_motion_time = None
last_status_update = 0
fps_calculated = 10  # Default FPS, updated dynamically

# Video writer settings
fourcc = cv2.VideoWriter_fourcc(*'h264')

def rotate_frame(frame, angle):
    """ Rotate a frame by the specified angle (clockwise). """
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame  # No rotation if angle == 0

def save_video(frames, fps, filename):
    """ Saves frames to a video file at the detected FPS, applying rotation if needed. """
    if not frames:
        return

    # Apply rotation to the first frame to get new dimensions
    first_frame = rotate_frame(frames[0], ROTATION_ANGLE)
    height, width, _ = first_frame.shape
    filepath = os.path.join(SAVE_DIRECTORY, filename)

    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    for frame in frames:
        frame = rotate_frame(frame, ROTATION_ANGLE)  # Rotate before saving
        out.write(frame)

    out.release()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Video saved: {filepath} at {fps:.2f} FPS")

# Open video stream
cap = cv2.VideoCapture(CAMERA_URL)
frame_times = []  # Track frame arrival times for FPS calculation

while True:
    start_time = time.time()  # Start timer for FPS calculation

    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Track timestamps to calculate FPS
    frame_times.append(start_time)
    frame_times = [t for t in frame_times if start_time - t < 1.0]  # Keep timestamps within 1 second
    fps_calculated = len(frame_times)  # Dynamic FPS measurement

    # Adjust buffer size dynamically based on detected FPS
    frame_buffer = deque(maxlen=BUFFER_SECONDS * fps_calculated)

    # Resize to 640x480 and convert to grayscale
    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Add frame to buffer
    frame_buffer.append(frame_resized)

    # Initialize reference frame
    if reference_frame is None:
        reference_frame = gray_blurred
        continue

    # Compute absolute difference
    frame_diff = cv2.absdiff(reference_frame, gray_blurred)
    thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Count changed pixels
    motion_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    motion_ratio = motion_pixels / total_pixels  # Percentage of changed pixels

    # Check for motion
    if motion_ratio > MIN_MOTION_RATIO:
        last_motion_time = time.time()  # Update last detected motion time
        if not recording:
            recording = True
            record_start_time = time.time()  # Mark when recording starts
            motion_start_time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(record_start_time))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚨 Motion detected! Recording started.")
            recording_frames = list(frame_buffer)  # Save last BUFFER_SECONDS

    # If recording, continue adding frames
    if recording:
        recording_frames.append(frame_resized)

        # Stop recording if no motion for POST_MOTION_SECONDS or max time reached
        if time.time() - last_motion_time > POST_MOTION_SECONDS or time.time() - record_start_time > MAX_RECORD_SECONDS:
            filename = f"{CAMERA_NAME}_motion_{motion_start_time_str}.mp4"
            save_video(recording_frames, fps_calculated, filename)
            recording = False
            recording_frames = []  # Clear recorded frames buffer
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🟢 Waiting for new motion...")

    # Print status updates at most once per second
    if time.time() - last_status_update > 1:
        status_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STATUS: " \
                     f"{'📹 Recording' if recording else '🟢 Monitoring'} | " \
                     f"Motion Ratio: {motion_ratio:.2%} | FPS: {fps_calculated:.2f}"
        print(status_msg)
        last_status_update = time.time()

cap.release()
cv2.destroyAllWindows()
