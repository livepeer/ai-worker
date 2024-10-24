import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from typing import AsyncGenerator

from PIL import Image

from .process import PipelineProcess

fps_log_interval = 10


class PipelineStreamer(ABC):
    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.process = None
        self.last_params: dict | None = None
        self.last_params_time = 0.0
        self.restart_count = 0

    def start(self):
        self.process = PipelineProcess.start(self.pipeline)
        self.ingress_task = asyncio.create_task(self.run_ingress_loop(self.process.done))
        self.egress_task = asyncio.create_task(self.run_egress_loop(self.process.done))
        self.monitor_task = asyncio.create_task(self.monitor_loop(self.process.done))

    async def stop(self):
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

    async def restart(self):
        try:
            await self.stop()
            self.start()
            self.restart_count += 1
            logging.info(
                f"PipelineProcess restarted. Restart count: {self.restart_count}"
            )

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
            if not self.process:
                return

            current_time = time.time()
            time_since_last_input = current_time - self.process.last_input_time
            time_since_last_output = current_time - self.process.last_output_time
            time_since_start = current_time - start_time
            time_since_last_params = current_time - self.last_params_time
            time_since_reload = min(time_since_last_params, time_since_start)

            if time_since_last_input > 5:
                # nothing to do if we're not sending inputs
                continue

            active_after_reload = (
                time_since_last_output < (time_since_reload - 1)
                or self.pipeline == "liveportrait"  # liveportrait gets stuck on load
            )
            stopped_recently = (
                active_after_reload
                and time_since_last_output > 5
                and time_since_last_output < 60
            )
            gone_stale = time_since_last_output > 60 and time_since_reload > 60
            if stopped_recently or gone_stale:
                logging.warning(
                    "No output received while inputs are being sent. Restarting process."
                )
                await self.restart()
                return

    async def run_ingress_loop(self, done: Event):
        frame_count = 0
        start_time = time.time()
        async for frame in self.ingress_loop(done):
            if done.is_set() or not self.process:
                return
            self.process.send_input(frame)

            # Increment frame count and measure FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= fps_log_interval:
                fps = frame_count / elapsed_time
                logging.info(f"Input FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

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

        await self.egress_loop(gen_output_frames())

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
