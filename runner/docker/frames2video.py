from flask import Flask, Response, request, jsonify
import os
import time
import threading
import subprocess
import sys
from collections import defaultdict

app = Flask(__name__)

# input and output framerate for ffmpeg
FRAMERATE=24
GOP_SECS=3

# Structure to hold per-stream data
class StreamData:
    def __init__(self):
        self.out_pipe = None # main image pipe to feed ffmpeg output
        self.images = [] # A list to store image byte data
        self.count = 0
        self.eof = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.ffmpeg_in_thread = None
        self.ffmpeg_out_thread = None
        self.write_images_thread = None
        self.read_fd = None
        self.write_fd = None

# Dictionary to store stream-specific data
streams = defaultdict(StreamData)

def append_to_stream(stream_name, image_data, idx):
    stream_data = streams[stream_name]

    with stream_data.condition:
        if stream_data.eof:
            print(f"[Append-{stream_name}] Dropping frame {idx} due to EOF")
            return
        if stream_data.count != idx:
            # just a bit of double checking in case images arrive out of order
            # TODO handle this better, eg reorder as necessary
            print(f"[Append-{stream_name}] Mismatch image indices! Received {idx} expected {stream_data.count}")
        stream_data.images.append({'idx': idx, 'image': image_data, 'time': time.time()})
        stream_data.count += 1
        # print(f"[Append-{stream_name}] Appended image data: {idx}")
        stream_data.condition.notify_all()  # Signal all waiting threads

def run_ffmpeg_in(stream_name):
    # Prepare and execute a FFmpeg subprocess for INPUT
    input_url = f"rtmp://localhost/{stream_name}"
    output_files = f"http://localhost:3080/upload_frame/{stream_name}/%d.jpg"
    ffmpeg_command = ["ffmpeg", "-loglevel", "warning", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", input_url, "-an", "-vf", f"scale_cuda=w=512:h=512:force_original_aspect_ratio=decrease:force_divisible_by=2,hwdownload,format=nv12,fps={FRAMERATE}", "-c:v", "mjpeg", "-start_number", "0", "-q:v", "1", output_files]
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Capture real-time output from stderr
    for line in process.stderr:
        print(line, end="")
    process.wait()
    if process.returncode == 0:
        print("\n[FFmpeg-in-{stream_name}] command completed successfully")
    else:
        print("\n[FFmpeg-in-{stream_name}] Error executing command")

def run_ffmpeg_out(fd, stream_name):
    # Prepare and execute a FFmpeg subprocess for OUTPUT
    output_url = f"rtmp://localhost/{stream_name}/out"
    input_files = f"pipe:{fd}"
    ffmpeg_command = ["ffmpeg", "-loglevel", "warning", "-f", "image2pipe", "-framerate", f"{FRAMERATE}", "-i", input_files, "-c:v", "h264_nvenc", "-bf", "0", "-g", f"{GOP_SECS*FRAMERATE}", "-preset","p1", "-tune", "ull", "-f", "flv", output_url]
    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(fd,), text=True)

    # Capture real-time output from stderr
    for line in process.stderr:
        print(line, end="")
    process.wait()
    if process.returncode == 0:
        print("\n[FFmpeg-out-{stream_name}] command completed successfully")
    else:
        print("\n[FFmpeg-out-{stream_name}] Error executing command")

def write_images(stream_name):
    # Writes read images to a pipe for ffmpeg
    stream_data = streams[stream_name]

    # Track ffmpeg image counts separately from received image indices
    # That way if we have to drop images from the queue, we can still
    # maintain continuous numbering for ffmpeg
    image_count = 0

    while True:
        with stream_data.condition:
            # Wait until there are new data in the list or EOF is signaled
            stream_data.condition.wait_for(lambda: stream_data.images or stream_data.eof)

            # If EOF is signaled and there are no new images, exit
            if stream_data.eof and not stream_data.images:
                print(f"[Processor-{stream_name}] EOF received. Stopping.")
                break

            # Copy the list and clear the original to minimize locking
            out_pipe = stream_data.out_pipe
            if out_pipe is None:
                if stream_data.eof:
                    print(f"[Processor-{stream_name}] EOF received. Quitting and dropping {len(stream_data.images)} images.")
                    break
                print(f"[Processor-{stream_name}] Early initialization before pipe was ready, skipping")
                time.sleep(1) # hack for now
                continue
            images_to_process = stream_data.images[:]
            stream_data.images.clear()

        # need to know if the queue is filling up faster than realtime
        if len(images_to_process) > 1:
            print(f"[Processor-{stream_name}] Queue: {len(images_to_process)}")

        # Process the copied list outside of the lock
        for image_data in images_to_process:
            idx = image_data['idx']
            if image_count != idx:
                # just a bit of double checking in case images arrive out of order
                print(f"[Processor-{stream_name}] Mismatch image indices! Received {idx} expected {image_count}")
            try:
                out_pipe.write(image_data['image'])
            except BrokenPipeError:
                print(f"[Processor-{stream_name} Pipe had no reader at {idx}! Bail out!")
            image_count += 1
            if False:
                print(f"[Processor-{stream_name} {idx} took {(time.time() - image_data['time'])*1000} ms")

def start_stream_threads(stream_name):
    stream_data = streams[stream_name]
    read_fd, write_fd = os.pipe()
    ffmpeg_in_thread = threading.Thread(target=run_ffmpeg_in, args=(stream_name,))
    ffmpeg_in_thread.start()
    ffmpeg_out_thread = threading.Thread(target=run_ffmpeg_out, args=(read_fd, stream_name))
    ffmpeg_out_thread.start()
    write_images_thread = threading.Thread(target=write_images, args=(stream_name,))
    write_images_thread.start()
    # TODO error handling
    out_pipe = os.fdopen(write_fd, "wb")
    with stream_data.lock:
        stream_data.ffmpeg_in_thread = ffmpeg_in_thread
        stream_data.ffmpeg_out_thread = ffmpeg_out_thread
        stream_data.write_images_thread = write_images_thread
        stream_data.out_pipe = out_pipe
        stream_data.write_fd = write_fd
        stream_data.read_fd = read_fd

@app.route('/unpublished', methods=['POST'])
def unpublished():
    # Parsing form data
    form_data = request.form  # For 'application/x-www-form-urlencoded'
    stream_name = form_data['stream']


    # Send EOF signal for this stream
    stream_data = streams[stream_name]
    with stream_data.condition:
        stream_data.eof = True
        ffmpeg_in_thread = stream_data.ffmpeg_in_thread
        ffmpeg_out_thread = stream_data.ffmpeg_out_thread
        write_images_thread = stream_data.write_images_thread
        count = stream_data.count

        if stream_data.read_fd is not None:
            os.close(stream_data.read_fd)
            stream_data.read_fd = None
        if stream_data.write_fd is not None:
            os.close(stream_data.write_fd)
            stream_data.write_fd = None

        print(f"[Unpublish-{stream_name}] EOF signal sent")
        stream_data.condition.notify_all()

    # Wait for ffmpeg INPUT thread to exit
    if ffmpeg_in_thread:
        print(f"[Unpublish-{stream_name}] waiting for ffmpeg INPUT")
        ffmpeg_in_thread.join()

    # Wait for write images thread to exit
    if write_images_thread:
        print(f"[Unpublish-{stream_name}] waiting for write images thread")
        write_images_thread.join()

    # Wait for ffmpeg OUTPUT thread to exit
    if ffmpeg_out_thread:
        print(f"[Unpublish-{stream_name}] waiting for ffmpeg OUTPUT")
        ffmpeg_out_thread.join()

    return jsonify({"response":"OK"})

@app.route('/ready', methods=['POST'])
def submit_form():
    # Parsing form data
    form_data = request.form  # For 'application/x-www-form-urlencoded'
    stream_name = form_data['stream']

    # don't try to process outbound publish
    if stream_name.endswith('/out'):
        return jsonify({"response": "OK"})

    # not confident this is safe across concurrent requests
    if stream_name in streams and not streams[stream_name].eof:
        print(f"Stream existed! OH NO we are stomping over something")
        # TODO dodo?
        return jsonify({"response":"most definitely NOT OK"})
    else:
        streams[stream_name] = StreamData()

    start_stream_threads(stream_name)
    return jsonify({"response":"OK"})

@app.route('/upload_frame/<stream_name>/<int:frame_number>.jpg', methods=['POST'])
def upload_frame(stream_name, frame_number):

    # Read the raw binary data sent by ffmpeg
    append_to_stream(stream_name, request.data, frame_number)

    return jsonify({"message": f"Frame '{stream_name}-{frame_number}' successfully uploaded"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3080)
