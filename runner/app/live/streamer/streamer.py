import asyncio
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from typing import AsyncGenerator

from PIL import Image

from .process import PipelineProcess

fps_log_interval = 10


class PipelineStreamer(ABC):
    def __init__(self, pipeline: str, input_timeout: int, params: dict):
        self.pipeline = pipeline
        self.params = params
        self.process = None
        self.last_params_time = 0.0
        self.restart_count = 0
        self.input_timeout = input_timeout  # 0 means disabled
        self.done_future = None

    def start(self):
        self.done_future = asyncio.get_running_loop().create_future()
        self._start_process()

    async def wait(self):
        if not self.done_future:
            raise RuntimeError("Streamer not started")
        return await self.done_future

    async def stop(self):
        await self._stop_process()
        if self.done_future and not self.done_future.done():
            self.done_future.set_result(None)

    def _start_process(self):
        if self.process:
            raise RuntimeError("PipelineProcess already started")

        self.process = PipelineProcess.start(self.pipeline, **self.params)
        self.ingress_task = asyncio.create_task(self.run_ingress_loop(self.process.done))
        self.egress_task = asyncio.create_task(self.run_egress_loop(self.process.done))
        self.monitor_task = asyncio.create_task(self.monitor_loop(self.process.done))

    async def _stop_process(self):
        if self.process:
            self.process.stop()
            self.process = None

        if self.ingress_task:
            self.ingress_task.cancel()
            self.ingress_task = None

        if self.egress_task:
            self.egress_task.cancel()
            self.egress_task = None

        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None

    async def _restart(self):
        try:
            # don't call the start/stop methods since those might be overridden by the concrete implementations
            await self._stop_process()
            self._start_process()
            self.restart_count += 1
            logging.info(
                f"PipelineProcess restarted. Restart count: {self.restart_count}"
            )
        except Exception as e:
            logging.error(f"Error restarting pipeline process: {e}")
            logging.error(f"Stack trace:\n{traceback.format_exc()}")
            os._exit(1)

    def update_params(self, params: dict):
        self.params = params
        self.last_params_time = time.time()
        if self.process:
            self.process.update_params(**params)

    async def monitor_loop(self, done: Event):
        start_time = time.time()
        while not done.is_set():
            await asyncio.sleep(2)
            if not self.process:
                return

            current_time = time.time()
            last_input_time = self.process.last_input_time or start_time
            last_output_time = self.process.last_output_time or start_time
            logging.debug(f"Monitoring process. last_input_time: {last_input_time}, last_output_time: {last_output_time} last_params_time: {self.last_params_time}")

            time_since_last_input = current_time - last_input_time
            time_since_last_output = current_time - last_output_time
            time_since_start = current_time - start_time
            time_since_last_params = current_time - self.last_params_time
            time_since_reload = min(time_since_last_params, time_since_start)

            if self.input_timeout > 0 and time_since_last_input > self.input_timeout:
                logging.info(f"Input stream stopped for {time_since_last_input} seconds. Shutting down...")
                await asyncio.create_task(self.stop())
                return

            gone_stale = (
                time_since_last_output > time_since_last_input
                and time_since_last_output > 60
                and time_since_reload > 60
            )
            if time_since_last_input > 5 and not gone_stale:
                # nothing to do if we're not sending inputs
                continue

            active_after_reload = time_since_last_output < (time_since_reload - 1)
            stopped_recently = (
                time_since_last_output > 5
                if self.pipeline == "liveportrait" # liveportrait loads very quick but gets stuck too often
                else active_after_reload and time_since_last_output > 5 and time_since_last_output < 60
            )
            if stopped_recently or gone_stale:
                logging.warning(
                    "No output received while inputs are being sent. Restarting process."
                )
                await self._restart()
                return

    async def run_ingress_loop(self, done: Event):
        frame_count = 0
        start_time = time.time()
        try:
            async for frame in self.ingress_loop(done):
                if done.is_set() or not self.process:
                    return

                # crop the max square from the center of the image and scale to 512x512
                # most models expect this size especially when using tensorrt
                width, height = frame.size
                if width != height:
                    square_size = min(width, height)
                    frame = frame.crop((width // 2 - square_size // 2, height // 2 - square_size // 2, width // 2 + square_size // 2, height // 2 + square_size // 2))
                if frame.size != (512, 512):
                    frame = frame.resize((512, 512))

                logging.debug(f"Sending input frame. Scaled from {width}x{height} to 512x512")
                self.process.send_input(frame)

                # Increment frame count and measure FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= fps_log_interval:
                    fps = frame_count / elapsed_time
                    logging.info(f"Input FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()
            # automatically stop the streamer when the ingress ends cleanly
            await self.stop()
        except Exception:
            logging.error("Error running ingress loop", exc_info=True)
            await self._restart()

    async def run_egress_loop(self, done: Event):
        async def gen_output_frames() -> AsyncGenerator[Image.Image, None]:
            frame_count = 0
            start_time = time.time()
            while not done.is_set() and self.process:
                output_image = await self.process.recv_output()
                if not output_image:
                    break
                logging.debug(
                    f"Output image received out_width: {output_image.width}, out_height: {output_image.height}"
                )

                yield output_image

                # Increment frame count and measure FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= fps_log_interval:
                    fps = frame_count / elapsed_time
                    logging.info(f"Output FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()

        try:
            await self.egress_loop(gen_output_frames())
            # automatically stop the streamer when the egress ends cleanly
            await self.stop()
        except Exception:
            logging.error("Error running egress loop", exc_info=True)
            await self._restart()

    @abstractmethod
    async def ingress_loop(self, done: Event) -> AsyncGenerator[Image.Image, None]:
        """Generator that yields the ingress frames."""
        if False:
            yield Image.new('RGB', (1, 1)) # dummy yield for linter to see this is a generator
        pass

    @abstractmethod
    async def egress_loop(self, output_frames: AsyncGenerator[Image.Image, None]):
        """Consumes generated frames and processes them."""
        pass
