import json
import base64
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os

def extract_base64_string(data_url):
    # Remove the 'data:image/png;base64,' prefix
    base64_str = data_url.split(',', 1)[1]
    return base64_str

def add_padding(base64_string):
    # Add padding if necessary
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += '=' * (4 - missing_padding)
    return base64_string

def convert_base64_to_image(base64_string, output_path):
    try:
        # Add padding to the base64 string
        base64_string = add_padding(base64_string)
        
        # Decode the base64 string to bytes
        image_data = base64.b64decode(base64_string)
        
        # Convert bytes to an image
        image = Image.open(BytesIO(image_data))
        
        # Save the image to a file
        image.save(output_path)
        print(f"Image saved successfully to {output_path}")
    except (base64.binascii.Error, UnidentifiedImageError) as e:
        print(f"Failed to decode and save image: {e}")

def extract_and_convert_images(json_file_path, output_dir):
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the base64 strings from the URLs
        if 'images' in data:
            for idx, image_info in enumerate(data['images']):
                if 'url' in image_info:
                    data_url = image_info['url']
                    base64_string = extract_base64_string(data_url)
                    
                    output_image_path = os.path.join(output_dir, f'image_{idx}.png')
                    convert_base64_to_image(base64_string, output_image_path)
        else:
            print("Invalid JSON schema or missing 'url' field")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON file: {e}")

# Example usage
json_file_path = 'G:/ai-models/models/response_1722196176417.json'       # Path to your JSON file
output_image_path = 'G:/ai-models/models/output_image.jpg'  # Path to save the output image
extract_and_convert_images(json_file_path, output_image_path)
