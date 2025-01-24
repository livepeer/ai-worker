import aiohttp
import asyncio
import logging
import os
import threading
import subprocess

from .trickle_subscriber import TrickleSubscriber
from .trickle_publisher import TricklePublisher
from .decoder import decode_av
from . import segmenter

# target framerate
FRAMERATE=segmenter.FRAMERATE

# TODO make this better configurable
GPU=segmenter.GPU

async def run_subscribe(subscribe_url: str, image_callback):
    # TODO add some pre-processing parameters, eg image size
    try:
        read_fd, write_fd = os.pipe()
        parse_task = asyncio.create_task(decode_in(read_fd, image_callback))
        subscribe_task = asyncio.create_task(subscribe(subscribe_url, await AsyncifyFdWriter(write_fd)))
        await asyncio.gather(subscribe_task, parse_task)
        logging.info("run_subscribe complete")
    except Exception as e:
        logging.error(f"preprocess got error {e}", e)
        raise e

async def subscribe(subscribe_url, out_pipe):
    async with TrickleSubscriber(url=subscribe_url) as subscriber:
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

async def AsyncifyFdWriter(write_fd):
    loop = asyncio.get_event_loop()
    write_protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader(), loop=loop)
    write_transport, _ = await loop.connect_write_pipe( lambda: write_protocol, os.fdopen(write_fd, 'wb'))
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)
    return writer

async def decode_in(in_pipe, frame_callback):
    def decode_runner():
        try:
            decode_av(f"pipe:{in_pipe}", frame_callback)
        except Exception as e:
            logging.error(f"Decoding error {e}", exc_info=True)
        finally:
            os.close(in_pipe)
            logging.info("Decoding finished")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, decode_runner)

def feed_ffmpeg(ffmpeg_fd, image_generator):
    while True:
        image = image_generator()
        if image is None:
            logging.info("Image generator empty, leaving feed_ffmpeg")
            break
        os.write(ffmpeg_fd, image)
    os.close(ffmpeg_fd)

async def run_publish(publish_url: str, image_generator):
    publisher = None
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
        if publisher:
            await publisher.close()
