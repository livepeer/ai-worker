import cv2
import numpy as np
import tempfile
import os
from io import BytesIO

def extract_frames_from_video(video_data, is_file_path=True) -> np.ndarray:
    """
    Extract frames from a video file or in-memory video data and return them as a NumPy array.
    
    Args:
        video_data (str or BytesIO): Path to the input video file or in-memory video data.
        is_file_path (bool): Indicates if video_data is a file path (True) or in-memory data (False).
    
    Returns:
        np.ndarray: Array of frames with shape (num_frames, height, width, channels).
    """
    if is_file_path:
        # Handle file-based video input
        video_capture = cv2.VideoCapture(video_data)
    else:
        # Handle in-memory video input
        # Create a temporary file to store in-memory video data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_data.getvalue())
            temp_file_path = temp_file.name
        
        # Open the temporary video file
        video_capture = cv2.VideoCapture(temp_file_path)

    if not video_capture.isOpened():
        raise ValueError("Error opening video data")

    frames = []
    success, frame = video_capture.read()
    
    while success:
        frames.append(frame)
        success, frame = video_capture.read()
    
    video_capture.release()

    # Delete the temporary file if it was created
    if not is_file_path:
        os.remove(temp_file_path)
    
    # Convert list of frames to a NumPy array
    frames_array = np.array(frames)
    print(f"Extracted {frames_array.shape[0]} frames from video in shape of {frames_array.shape}")
    
    return frames_array

# Example usage
if __name__ == "__main__":
    # File path example
    video_file_path = "C:/Users/ganes/Desktop/Generated videos/output.mp4"


    # In-memory video example
    with open(video_file_path, "rb") as f:
        video_data = BytesIO(f.read())
    frames_array_from_memory = extract_frames_from_video(video_data, is_file_path=False)
