from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Directory where uploaded frames will be saved
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory list (acting as a queue) for storing frame filenames
frame_queue = []

@app.route('/upload_frame/<stream_name>/<int:frame_number>.jpg', methods=['POST'])
def upload_frame(stream_name, frame_number):
    # Generate a unique filename for each incoming frame
    frame_filename = f"{stream_name}_{frame_number:04d}.jpg"

    # Read the raw binary data sent by ffmpeg
    try:
        with open(os.path.join(UPLOAD_FOLDER, frame_filename), 'wb') as f:
            f.write(request.data)  # Save the binary data to disk
    except Exception as e:
        return jsonify({"error": f"Failed to save frame: {str(e)}"}), 500

    # Add the frame name to the in-memory queue
    print(frame_filename)
    frame_queue.append(frame_filename)

    return jsonify({"message": f"Frame '{frame_filename}' successfully uploaded",
                    "queue": frame_queue}), 200

@app.route('/queue', methods=['GET'])
def get_queue():
    return jsonify({"queue": frame_queue}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3080)
