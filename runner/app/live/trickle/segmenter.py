import errno
import time
import logging
import os
import select
import string
import subprocess
import sys
import random
import threading
from datetime import datetime

# Constants and initial values
READ_TIMEOUT = 90
SLEEP_INTERVAL = 0.05

# TODO make this better configurable
FRAMERATE=24
GOP_SECS=3
GPU=False

def create_named_pipe(pattern, pipe_index):
    pipe_name = pattern % pipe_index
    try:
        os.mkfifo(pipe_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return pipe_name

def remove_named_pipe(pipe_name):
    try:
        os.remove(pipe_name)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def ffmpeg_cmd(out_pattern):

    if GPU:
        cmd = [
        'ffmpeg',
	    '-loglevel', 'warning',
        '-use_wallclock_as_timestamps', '1',
        '-f', 'image2pipe',
        '-i', 'pipe:0', # stdin
        '-c:v', 'h264_nvenc',
        '-bf', '0', # disable bframes for webrtc
        '-force_key_frames', f'expr:gte(t,n_forced*{GOP_SECS})',
        '-preset', 'p1',
        '-tune', 'ull',
        '-f', 'segment',
        out_pattern
        ]
    else:
        cmd = [
        'ffmpeg',
	    '-loglevel', 'warning',
        '-use_wallclock_as_timestamps', '1',
        '-f', 'image2pipe',
        '-i', 'pipe:0', # stdin
        '-c:v', 'libx264',
        '-bf', '0', # disable bframes for webrtc
        '-force_key_frames', f'expr:gte(t,n_forced*{GOP_SECS})',
        '-preset', 'superfast',
        '-tune', 'zerolatency',
        '-f', 'segment',
        out_pattern
        ]

    logging.info(f"ffmpeg (output) {cmd}")
    return cmd


def read_from_pipe(pipe_name, callback, ffmpeg_proc):
    fd = os.open(pipe_name, os.O_RDONLY | os.O_NONBLOCK)

    # Polling to check if the pipe is ready
    poller = select.poll()
    poller.register(fd, select.POLLIN)

    start_time = time.time()
    while True:
        # Wait for the pipe to become ready for reading
        events = poller.poll(1000 * SLEEP_INTERVAL)

        # If the pipe is ready, switch to blocking mode and read
        if events:
            os.set_blocking(fd, True)
            break

        # Check if ffmpeg has exited after polling
        if ffmpeg_proc.poll() is not None:
            logging.info(f"FFmpeg process has exited while waiting for pipe {pipe_name}")
            os.close(fd)
            return False


        # Check if we've exceeded the timeout
        if time.time() - start_time > READ_TIMEOUT:
            logging.info(f"Timeout waiting for pipe {pipe_name}")
            os.close(fd)
            return False

        # Sleep briefly before checking again
        time.sleep(SLEEP_INTERVAL)

    # Now that the pipe is ready, invoke the callback
    # fdopen will implcitly close the supplied fd
    with os.fdopen(fd, 'rb', buffering=0) as pipe_fd:
        callback(pipe_fd, pipe_name)

    remove_named_pipe(pipe_name)
    return True

def segment_reading_process(in_fd, callback):
    pipe_index = 0
    out_pattern = generate_random_string() + "-%d.ts"

    # Start by creating the first two named pipes
    current_pipe = create_named_pipe(out_pattern, pipe_index)
    next_pipe = create_named_pipe(out_pattern, pipe_index + 1)

    # Launch FFmpeg process with stdin, stdout, and stderr as pipes
    proc = subprocess.Popen(
        ffmpeg_cmd(out_pattern),
        stdin=in_fd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logging.debug("segment_reading_process: ffmpeg started")

    # Create a thread to handle stderr redirection
    thread = threading.Thread(target=print_proc, args=(proc.stdout,))
    thread.start()

    try:
        while True:
            # Read from the current pipe, exit the loop if there's a timeout or ffmpeg exit
            if not read_from_pipe(current_pipe, callback, proc):
                logging.info("Exiting ffmpeg (output) due to timeout or process exit.")
                break

            # Move to the next pipes in the sequence
            pipe_index += 1
            current_pipe = next_pipe

            # Create the new next pipe in the sequence
            next_pipe = create_named_pipe(out_pattern, pipe_index + 1)

    except Exception as e:
        logging.info(f"FFmpeg (output) error : {e} - {current_pipe}")

    finally:
        os.close(in_fd)
        logging.info("awaitng ffmpeg (output)")
        proc.wait()
        logging.info("proc complete ffmpeg (output)")
        thread.join()
        logging.info("ffmpeg (output) complete")

        # Cleanup remaining pipes
        remove_named_pipe(current_pipe)
        remove_named_pipe(next_pipe)

def print_proc(f):
    """Reads stderr from a subprocess and writes it to sys.stderr."""
    for line in iter(f.readline, b''):
        sys.stderr.write(line.decode())

def generate_random_string():
    """Generates a random string of length 5."""
    length=5
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))
