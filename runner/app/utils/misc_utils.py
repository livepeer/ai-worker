import cv2
import numpy as np
from PIL import Image

def fast_image_resize(frame, width=512, height=512):
    if frame.size == (width, height):
        return frame
    
    frame_array = np.array(frame)
    og_height, og_width = frame_array.shape[:2]

    if width == height and og_width != og_height:
        square_size = min(og_width, og_height)
        start_x = og_width // 2 - square_size // 2
        start_y = og_height // 2 - square_size // 2
        frame_array = frame_array[start_y:start_y+square_size, start_x:start_x+square_size]

    # Resize using cv2 (much faster than PIL)
    if frame_array.shape[:2] != (height, width):
        frame_array = cv2.resize(frame_array, (height, width))
    return Image.fromarray(frame_array)