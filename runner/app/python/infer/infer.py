import argparse
import asyncio
import logging
import io
import signal
import time
import traceback
import sys
from typing import List

import zmq.asyncio
from PIL import Image
from aiohttp import web
import os
import tempfile
import hashlib
from aiohttp import BodyPartReader
import mimetypes

from pipelines import load_pipeline
import multiprocessing as mp
from multiprocessing.synchronize import Event
import queue
fps_log_interval = 10

TEMP_SUBDIR = 'infer_temp'
MAX_FILE_AGE = 86400 # 1 day

def to_jpeg_bytes(frame: Image.Image):
    buffer = io.BytesIO()
    frame.save(buffer, format='JPEG')
    bytes = buffer.getvalue()
    buffer.close()
    return bytes

def from_jpeg_bytes(frame_bytes: bytes):
    image = Image.open(io.BytesIO(frame_bytes))
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    return image

class PipelineProcess:
    @staticmethod
    def start(pipeline_name: str):
        instance = PipelineProcess(pipeline_name)
        instance.process.start()
        return instance

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.ctx = mp.get_context('spawn')

        self.input_queue = self.ctx.Queue(maxsize=5)
        self.last_input_time = 0
        self.output_queue = self.ctx.Queue()
        self.last_output_time = 0
        self.param_update_queue = self.ctx.Queue()

        self.done = self.ctx.Event()
        self.process = self.ctx.Process(target=self.process_loop, args=())

    def stop(self):
        self.done.set()
        if self.process.is_alive():
            logging.info(f"Terminating pipeline process")
            self.process.terminate()
            try:
                self.process.join(timeout=5)
            except Exception as e:
                logging.error(f"Killing process due to join error: {e}")
                self.process.kill()

    def is_done(self):
        return self.done.is_set()

    def send_input(self, frame: Image.Image):
        while not self.is_done():
            try:
                self.input_queue.put_nowait(frame)
                self.last_input_time = time.time()
                break
            except queue.Full:
                try:
                    # remove oldest frame from queue to add new one
                    self.input_queue.get_nowait()
                except queue.Empty:
                    continue

    async def recv_output(self):
        # we cannot do a long get with timeout as that would block the asyncio
        # event loop, so we loop with nowait and sleep async instead.
        while not self.is_done():
            try:
                output = self.output_queue.get_nowait()
                self.last_output_time = time.time()
                return output
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue
        return None

    def process_loop(self):
        logging.basicConfig(level=logging.INFO)
        try:
            params = {}
            try:
                params = self.param_update_queue.get_nowait()
            except queue.Empty:
                pass
            except Exception as e:
                logging.error(f"Error getting params: {e}")

            try:
                pipeline = load_pipeline(self.pipeline_name, **params)
                logging.info("Pipeline loaded successfully")
            except Exception as e:
                logging.error(f"Error loading pipeline: {e}")
                pipeline = load_pipeline(self.pipeline_name)
                logging.info("Pipeline loaded with default params")

            while not self.is_done():
                if not self.param_update_queue.empty():
                    params = self.param_update_queue.get_nowait()
                    try:
                        pipeline.update_params(**params)
                        logging.info(f"Updated params: {params}")
                    except Exception as e:
                        logging.error(f"Error updating params: {e}")

                try:
                    input_image = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    logging.debug(f"Input queue empty")
                    continue

                try:
                    output_image = pipeline.process_frame(input_image)
                    self.output_queue.put(output_image)
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
        except Exception as e:
            logging.error(f"Error in process run method: {e}")

class SocketHandler:
    def __init__(self, input_socket: zmq.asyncio.Socket, output_socket: zmq.asyncio.Socket, pipeline: str):
        self.input_socket = input_socket
        self.output_socket = output_socket
        self.pipeline = pipeline
        self.process = None
        self.last_params = None
        self.last_params_time = 0
        self.restart_count = 0

    def start(self):
        self.process = PipelineProcess.start(self.pipeline)
        self.input_task = asyncio.create_task(self.input_loop(self.process.done))
        self.output_task = asyncio.create_task(self.output_loop(self.process.done))
        self.monitor_task = asyncio.create_task(self.monitor_loop(self.process.done))

    async def stop(self):
        if self.process:
            self.process.stop()
            self.process = None

        if self.input_task:
            self.input_task.cancel()
            self.input_task = None

        if self.output_task:
            self.output_task.cancel()
            self.output_task = None

        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None

    async def restart(self):
        try:
            await self.stop()
            self.start()
            self.restart_count += 1
            logging.info(f"PipelineProcess restarted. Restart count: {self.restart_count}")

            if self.last_params:
                self.update_params(self.last_params)
        except Exception as e:
            logging.error(f"Error restarting pipeline process: {e}")
            logging.error(f"Stack trace:\n{traceback.format_exc()}")

    def update_params(self, params: dict):
        self.last_params = params
        self.last_params_time = time.time()
        if self.process:
            self.process.param_update_queue.put(params)

    async def monitor_loop(self, done: Event):
        start_time = time.time()
        while not done.is_set():
            await asyncio.sleep(2)
            current_time = time.time()
            time_since_last_input = current_time - self.process.last_input_time
            time_since_last_output = current_time - self.process.last_output_time
            time_since_start = current_time - start_time
            time_since_last_params = current_time - self.last_params_time
            time_since_reload = min(time_since_last_params, time_since_start)

            if time_since_last_input > 5:
                # nothing to do if we're not sending inputs
                continue

            stopped_recently = time_since_last_output < (time_since_reload + 1) and time_since_last_output > 5 and time_since_last_output < 60
            gone_stale = time_since_last_output > 60 and time_since_reload > 60
            if stopped_recently or gone_stale:
                logging.warning("No output received while inputs are being sent. Restarting process.")
                await self.restart()
                return

    async def input_loop(self, done: Event):
        frame_count = 0
        start_time = time.time()
        while not done.is_set():
            frame_bytes = await self.input_socket.recv()
            frame = from_jpeg_bytes(frame_bytes)

            self.process.send_input(frame)

            # Increment frame count and measure FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= fps_log_interval:
                fps = frame_count / elapsed_time
                logging.info(f"Input FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

    async def output_loop(self, done: Event):
        frame_count = 0
        start_time = time.time()
        while not done.is_set():
            output_image = await self.process.recv_output()
            if not output_image:
                break
            logging.debug(f"Output image received out_width: {output_image.width}, out_height: {output_image.height}")

            await self.output_socket.send(to_jpeg_bytes(output_image))

            # Increment frame count and measure FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= fps_log_interval:
                fps = frame_count / elapsed_time
                logging.info(f"Output FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()


def cleanup_old_files(temp_dir):
    current_time = time.time()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > MAX_FILE_AGE:
                os.remove(file_path)
                logging.info(f"Removed old file: {file_path}")

async def handle_params_update(request):
    try:
        params = {}
        temp_dir = os.path.join(tempfile.gettempdir(), TEMP_SUBDIR)
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_old_files(temp_dir)

        if request.content_type.startswith('application/json'):
            params = await request.json()
        elif request.content_type.startswith('multipart/'):
            reader = await request.multipart()

            async for part in reader:
                if part.name == 'params':
                    params.update(await part.json())
                elif isinstance(part, BodyPartReader):
                    content = await part.read()

                    file_hash = hashlib.md5(content).hexdigest()
                    content_type = part.headers.get('Content-Type', 'application/octet-stream')
                    ext = mimetypes.guess_extension(content_type) or ''
                    new_filename = f"{file_hash}{ext}"

                    file_path = os.path.join(temp_dir, new_filename)
                    with open(file_path, 'wb') as f:
                        f.write(content)

                    params[part.name] = file_path
        else:
            raise ValueError(f"Unknown content type: {request.content_type}")

        request.app['handler'].update_params(params)
        return web.Response(text="Params updated successfully")
    except Exception as e:
        logging.error(f"Error updating params: {e}")
        return web.Response(text=f"Error updating params: {str(e)}", status=400)

async def start_http_server(handler: SocketHandler, port: int):
    app = web.Application()
    app['handler'] = handler
    app.router.add_post('/api/params', handle_params_update)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logging.info(f"HTTP server started on port {port}")
    return runner

async def main(http_port: int, input_address: str, output_address: str, pipeline: str):
    context = zmq.asyncio.Context()

    input_socket = context.socket(zmq.SUB)
    input_socket.connect(input_address)
    input_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages
    input_socket.set_hwm(10)

    output_socket = context.socket(zmq.PUB)
    output_socket.connect(output_address)
    output_socket.set_hwm(10)

    handler = SocketHandler(input_socket, output_socket, pipeline)
    runner: web.AppRunner
    try:
        handler.start()
        runner = await start_http_server(handler, http_port)
    except Exception as e:
        logging.error(f"Error starting socket handler or HTTP server: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e

    await block_until_signal([signal.SIGINT, signal.SIGTERM])
    try:
        await runner.cleanup()
        await handler.stop()
    except Exception as e:
        logging.error(f"Error stopping room handler: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e

async def block_until_signal(sigs: List[signal.Signals]):
    loop = asyncio.get_running_loop()
    future: asyncio.Future[signal.Signals] = loop.create_future()

    def signal_handler(sig, _):
        logging.info(f"Received signal: {sig}")
        loop.call_soon_threadsafe(future.set_result, sig)
    for sig in sigs:
        signal.signal(sig, signal_handler)
    return await future

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer process to run the AI pipeline")
    parser.add_argument("--http-port", type=int, default=8888, help="Port for the HTTP server")
    parser.add_argument("--input-address", type=str, default="tcp://localhost:5555", help="Address for the input socket")
    parser.add_argument("--output-address", type=str, default="tcp://localhost:5556", help="Address for the output socket")
    parser.add_argument("--pipeline", type=str, default="streamkohaku", help="Pipeline to use")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        asyncio.run(main(args.http_port, args.input_address, args.output_address, args.pipeline))
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        logging.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        sys.exit(1)
