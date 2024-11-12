import aiohttp
import asyncio
import logging
import os
import threading
import subprocess

from .trickle_subscriber import TrickleSubscriber
from .trickle_publisher import TricklePublisher
from .jpeg_parser import JPEGStreamParser
from . import segmenter

# target framerate
FRAMERATE=segmenter.FRAMERATE

# TODO make this better configurable
GPU=segmenter.GPU

async def run_subscribe(subscribe_url: str, image_callback):
    # TODO add some pre-processing parameters, eg image size
    try:
        ffmpeg = await launch_ffmpeg()
        logging_task = asyncio.create_task(log_pipe_async(ffmpeg.stderr))
        subscribe_task = asyncio.create_task(subscribe(subscribe_url, ffmpeg.stdin))
        jpeg_task = asyncio.create_task(parse_jpegs(ffmpeg.stdout, image_callback))
        await asyncio.gather(ffmpeg.wait(), logging_task, subscribe_task, jpeg_task)
        logging.info("run_subscribe complete")
    except Exception as e:
        logging.error(f"preprocess got error {e}", e)
        raise e

async def subscribe(subscribe_url, out_pipe):
    subscriber = TrickleSubscriber(url=subscribe_url)
    logging.info(f"launching subscribe loop for {subscribe_url}")
    while True:
        segment = None
        try:
            segment = await subscriber.next()
            if not segment:
                break # complete
            while True:
                chunk = await segment.read()
                if not chunk:
                    break # end of segment
                out_pipe.write(chunk)
                await out_pipe.drain()
        except aiohttp.ClientError as e:
            logging.info(f"Failed to read segment - {e}")
            break # end of stream?
        except Exception as e:
            raise e
        finally:
            if segment:
                await segment.close()
            else:
                # stream is complete
                out_pipe.close()
                break

async def launch_ffmpeg():
    if GPU:
        ffmpeg_cmd = [
        'ffmpeg',
	    '-loglevel', 'warning',
	    '-hwaccel', 'cuda',
	    '-hwaccel_output_format', 'cuda',
        '-i', 'pipe:0',       # Read input from stdin
	    '-an',
	    '-vf', 'scale_cuda=w=512:h=512:force_original_aspect_ratio=decrease:force_divisible_by=2,hwdownload,format=nv12,fps={FRAMERATE}'
	    '-c:v', 'mjpeg',
	    '-start_number', '0',
	    '-q:v', '1',
        '-f', 'image2pipe',
        'pipe:1'              # Output to stdout
        ]
    else:
        ffmpeg_cmd = [
        'ffmpeg',
	    '-loglevel', 'warning',
        '-i', 'pipe:0',       # Read input from stdin
	    '-an',
	    '-vf', f'scale=w=512:h=512:force_original_aspect_ratio=decrease:force_divisible_by=2,fps={FRAMERATE}',
	    '-c:v', 'mjpeg',
	    '-start_number', '0',
	    '-q:v', '1',
        '-f', 'image2pipe',
        'pipe:1'              # Output to stdout
        ]

    logging.info(f"ffmpeg (input) {ffmpeg_cmd}")
    # Launch FFmpeg process with stdin, stdout, and stderr as pipes
    process = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    return process  # Return the process handle

async def log_pipe_async(pipe):
    """Reads from a pipe and logs each line."""
    while True:
        line = await pipe.readline()
        if not line:
            break  # Exit when the pipe is closed
        # Decode the binary line and log it
        logging.info(line.decode().strip())

async def parse_jpegs(in_pipe, image_callback):
    chunk_size = 32 * 1024 # read in 32kb chunks
    with JPEGStreamParser(image_callback) as parser:
        # TODO this does not work on asyncio streams - figure out how to
        #      disable os buffering on readsdisable buffering on reads
        #pipe = os.fdopen(in_pipe.fileno(), 'rb', buffering=0)
        while True:
            chunk = await in_pipe.read(chunk_size)
            if not chunk:
                break
            await parser.feed(chunk)

def feed_ffmpeg(ffmpeg_fd, image_generator):
    while True:
        image = image_generator()
        if image is None:
            logging.info("Image generator empty, leaving feed_ffmpeg")
            break
        os.write(ffmpeg_fd, image)
    os.close(ffmpeg_fd)

async def run_publish(publish_url: str, image_generator):
    try:
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")

        loop = asyncio.get_running_loop()
        async def callback(pipe_file, pipe_name):
            # trickle publish a segment with the contents of `pipe_file`
            async with await publisher.next() as segment:
                # convert pipe_fd into an asyncio friendly StreamReader
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                transport, _ = await loop.connect_read_pipe(lambda: protocol, pipe_file)
                while True:
                    sz = 32 * 1024 # read in chunks of 32KB
                    data = await reader.read(sz)
                    if not data:
                        break
                    await segment.write(data)
                transport.close()

        def sync_callback(pipe_fd, pipe_name):
            # asyncio.connect_read_pipe expects explicit fd close
            # so we have to manually read, detect eof, then close
            r, w = os.pipe()
            rf = os.fdopen(r, 'rb', buffering=0)
            future = asyncio.run_coroutine_threadsafe(callback(rf, pipe_name), loop)
            try:
                while True:
                    data = pipe_fd.read(32 * 1024)
                    if not data:
                        break
                    os.write(w, data)
                os.close(w) # streamreader is very sensitive about this
                future.result()  # This blocks in the thread until callback completes
                rf.close() # also closes the read end of the pipe
            # Ensure any exceptions in the coroutine are caught
            except Exception as e:
                logging.error(f"Error in sync_callback: {e}")

        ffmpeg_read_fd, ffmpeg_write_fd = os.pipe()
        segment_thread = threading.Thread(target=segmenter.segment_reading_process, args=(ffmpeg_read_fd, sync_callback))
        ffmpeg_feeder = threading.Thread(target=feed_ffmpeg, args=(ffmpeg_write_fd, image_generator))
        segment_thread.start()
        ffmpeg_feeder.start()
        logging.debug("run_publish: ffmpeg feeder and segmenter threads started")

        def joins():
            segment_thread.join()
            ffmpeg_feeder.join()
        await asyncio.to_thread(joins)
        logging.info("run_publish complete")

    except Exception as e:
        logging.error(f"postprocess got error {e}", e)
        raise e
    finally:
        await publisher.close()
