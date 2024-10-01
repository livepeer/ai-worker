import argparse
import asyncio
import logging
import os
import io
import signal
import time
import traceback
from typing import Callable, List

import watchdog.events
import watchdog.observers
import zmq.asyncio
from PIL import Image
from aiohttp import web

from transmorgrifiers import Transmorgrifier
import multiprocessing as mp
import queue
prompt_file = "./prompt.txt"
fps_log_interval = 10

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

class FileWatcher(watchdog.events.FileSystemEventHandler):
    def __init__(self, filename: str, callback: Callable[[str], None]):
        self.filename = filename
        self.callback = callback
        self.setup_observer()

    def setup_observer(self):
        self.load_file()
        observer = watchdog.observers.Observer()
        observer.schedule(self, path=os.path.dirname(self.filename), recursive=False)
        observer.start()

    def on_modified(self, event):
        if event.src_path == self.filename:
            self.load_file()

    def load_file(self):
        try:
            with open(self.filename, 'r') as f:
                contents = f.read().strip()
                if contents:
                    self.callback(contents)
        except FileNotFoundError:
            pass

def load_transmorgrifier(name: str, **params) -> Transmorgrifier:
    if name == "streamkohaku":
        from transmorgrifiers.streamkohaku import StreamKohaku
        return StreamKohaku(**params)
    elif name == "liveportrait":
        from transmorgrifiers.liveportrait import LivePortrait
        return LivePortrait(**params)
    raise ValueError(f"Unknown transmorgrifier: {name}")

class TransmorgrifierProcess:
    def __init__(self, transmorgrifier_name: str):
        self.transmorgrifier_name = transmorgrifier_name

        self.ctx = mp.get_context('spawn')
        self.input_queue = self.ctx.Queue(maxsize=5)
        self.output_queue = self.ctx.Queue()
        self.param_update_queue = self.ctx.Queue()
        self.done = self.ctx.Event()

        self.process = self.ctx.Process(target=self.process_loop, args=())

    def start(self):
        self.process.start()

    def stop(self):
        self.done.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()

    def send_input(self, frame: Image.Image):
        while not self.is_done():
            try:
                self.input_queue.put_nowait(frame)
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
                return self.output_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue
        return None

    def is_done(self):
        return self.done.is_set()

    def process_loop(self):
        logging.basicConfig(level=logging.INFO)
        try:
            try:
                params = self.param_update_queue.get_nowait()
            except queue.Empty:
                params = {}
            transmorgrifier = load_transmorgrifier(self.transmorgrifier_name, **params)
            logging.info("Transmorgrifier loaded successfully")

            while not self.is_done():
                if not self.param_update_queue.empty():
                    params = self.param_update_queue.get_nowait()
                    try:
                        transmorgrifier.update_params(**params)
                        logging.info(f"Updated params: {params}")
                    except Exception as e:
                        logging.error(f"Error updating params: {e}")

                try:
                    input_image = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    logging.debug(f"Input queue empty")
                    continue

                try:
                    output_image = transmorgrifier.process_frame(input_image)
                    self.output_queue.put(output_image)
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
        except Exception as e:
            logging.error(f"Error in process run method: {e}")

class SocketHandler:
    def __init__(self, input_socket: zmq.asyncio.Socket, output_socket: zmq.asyncio.Socket, pipeline: str):
        self.input_socket = input_socket
        self.output_socket = output_socket
        self.process = TransmorgrifierProcess(pipeline)
        self.last_prompt = None
        self.prompt_watcher = FileWatcher(prompt_file, self.set_prompt)

    def start(self):
        self.input_task = asyncio.create_task(self.input_loop())
        self.process.start()
        self.output_task = asyncio.create_task(self.output_loop())

    async def stop(self):
        if self.input_task:
            self.input_task.cancel()
            try:
                await self.input_task
            finally:
                self.input_task = None

        self.process.stop()

        if self.output_task:
            self.output_task.cancel()
            try:
                await self.output_task
            finally:
                self.output_task = None

    def set_prompt(self, prompt: str):
        if prompt != self.last_prompt:
            self.update_params({'prompt': prompt})
            logging.info(f"Prompt: {prompt}")

    def update_params(self, params: dict):
        self.last_prompt = params.get('prompt', None)
        self.process.param_update_queue.put(params)

    async def input_loop(self):
        frame_count = 0
        start_time = time.time()
        while not self.process.is_done():
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

    async def output_loop(self):
        frame_count = 0
        start_time = time.time()
        while True:
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

async def handle_params_update(request):
    try:
        params = await request.json()
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

    asyncio.run(main(args.http_port, args.input_address, args.output_address, args.pipeline))
