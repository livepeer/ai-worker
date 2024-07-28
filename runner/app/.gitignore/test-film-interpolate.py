import os
import requests

# Define the URL of the FastAPI application
URL = "http://localhost:8000/FILMPipeline"

# Test with two images
def test_with_two_images():
    image1_path = "G:/ai-models/models/all_interpolated_frames/round_003_frame_74.png"
    image2_path = "G:/ai-models/models/all_interpolated_frames/round_003_frame_36.png"

    with open(image1_path, "rb") as image1, open(image2_path, "rb") as image2:
        files = {
            "image1": ("image1.png", image1, "image/png"),
            "image2": ("image2.png", image2, "image/png"),
        }
        data = {
            "inter_frames": 2,
            "model_id": "film_net_fp16.pt"
        }
        response = requests.post(URL, files=files, data=data)

    print("Test with two images")
    print(response.status_code)
    print(response.json())

# Test with a directory of images
def test_with_image_directory():
    image_dir = "path/to/image_directory"

    data = {
        "inter_frames": 2,
        "model_path": "path/to/film_net_fp16.pt",
        "image_dir": image_dir
    }
    response = requests.post(URL, data=data)

    print("Test with image directory")
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    # Ensure that the FastAPI server is running before executing these tests
    test_with_two_images()
    test_with_image_directory()
