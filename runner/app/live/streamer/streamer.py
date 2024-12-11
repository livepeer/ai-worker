import asyncio
import logging
import os
import time
import traceback
import numpy as np
from multiprocessing.synchronize import Event
from typing import AsyncGenerator
from asyncio import Lock

import cv2
from PIL import Image
from pydantic import BaseModel

from .process import PipelineProcess
from .protocol.protocol import StreamProtocol

fps_log_interval = 10
status_report_interval = 10

class PipelineStatus(BaseModel):
    """Holds metrics for the pipeline streamer"""
    type: str = "status"
    pipeline: str
    start_time: float
    last_params_update_time: float | None = None
    last_params: dict | None = None
    last_params_hash: str | None = None

    input_fps: float = 0.0
    output_fps: float = 0.0
    last_input_time: float | None = None
    last_output_time: float | None = None

    restart_count: int = 0
    last_restart_time: float | None = None
    last_restart_logs: list[str] | None = None  # Will contain last N lines before restart
    last_error: str | None = None
    last_error_time: float | None = None

    def update_params(self, params: dict):
        self.last_params = params
        self.last_params_hash = str(hash(str(sorted(params.items()))))
        return self

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Convert all fields ending with _time to milliseconds
        for field, value in data.items():
            if field.endswith('_time'):
                data[field] = _timestamp_to_ms(value)
        return data


def _timestamp_to_ms(v: float | None) -> int | None:
    return int(v * 1000) if v is not None else None


class PipelineStreamer:
    def __init__(self, protocol: StreamProtocol, pipeline: str, input_timeout: int, params: dict):
        self.protocol = protocol
        self.pipeline = pipeline
        self.params = params
        self.process = None
        self.input_timeout = input_timeout  # 0 means disabled
        self.done_future = None
        self.status = PipelineStatus(pipeline=pipeline, start_time=time.time()).update_params(params)
        self.control_task = None
        self.report_status_task = None
        self.report_status_lock = Lock()

    async def start(self):
        self.done_future = asyncio.get_running_loop().create_future()
        self._start_process()
        await self.protocol.start()
        self.control_task = asyncio.create_task(self.run_control_loop())
        self.report_status_task = asyncio.create_task(self.report_status_loop())

    async def wait(self):
        if not self.done_future:
            raise RuntimeError("Streamer not started")
        return await self.done_future

    async def stop(self):
        try:
            if self.report_status_task:
                self.report_status_task.cancel()
                self.report_status_task = None
            if self.control_task:
                self.control_task.cancel()
                self.control_task = None
            await self.protocol.stop()
            await self._stop_process()
        except Exception:
            logging.error("Error stopping streamer", exc_info=True)
        finally:
            if self.done_future and not self.done_future.done():
                self.done_future.set_result(None)

    def _start_process(self):
        if self.process:
            raise RuntimeError("PipelineProcess already started")

        self.process = PipelineProcess.start(self.pipeline, self.params)
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
            # Capture logs before stopping the process
            restart_logs = self.process.get_recent_logs()
            last_error = self.process.get_last_error()

            # don't call the full start/stop methods since we don't want to restart the protocol
            await self._stop_process()
            self._start_process()
            self.status.restart_count += 1
            self.status.last_restart_time = time.time()
            self.status.last_restart_logs = restart_logs
            if last_error:
                self.status.last_error = last_error

            await self._emit_monitoring_event({
                "type": "restart",
                "pipeline": self.pipeline,
                "restart_count": self.status.restart_count,
                "restart_time": self.status.last_restart_time,
                "restart_logs": restart_logs,
                "last_error": last_error
            })

            logging.info(
                f"PipelineProcess restarted. Restart count: {self.status.restart_count}"
            )
        except Exception:
            logging.error(f"Error restarting pipeline process", exc_info=True)
            os._exit(1)

    async def update_params(self, params: dict):
        self.params = params
        if self.process:
            self.process.update_params(params)
        self.status.last_params_update_time = time.time()
        self.status.update_params(params)

        await self._emit_monitoring_event({
            "type": "params_update",
            "pipeline": self.pipeline,
            "params": params,
            "params_hash": self.status.last_params_hash,
            "update_time": self.status.last_params_update_time
        })

    async def report_status_loop(self):
        next_report = time.time() + status_report_interval
        while not self.done_future.done():
            current_time = time.time()
            if next_report <= current_time:
                # If we lost track of the next report time, just report immediately
                next_report = current_time + status_report_interval
            else:
                await asyncio.sleep(next_report - current_time)
                next_report += status_report_interval

            event = self.status.model_dump()
            # Clear the large transient fields after reporting them once
            self.status.last_params = None
            self.status.last_restart_logs = None
            await self._emit_monitoring_event(event)

    async def _emit_monitoring_event(self, event: dict):
        """Protected method to emit monitoring event with lock"""
        event["timestamp"] = _timestamp_to_ms(time.time())
        async with self.report_status_lock:
            try:
                await self.protocol.emit_monitoring_event(event)
            except Exception as e:
                logging.error(f"Failed to emit monitoring event: {e}")

    async def monitor_loop(self, done: Event):
        start_time = time.time()
        while not done.is_set():
            await asyncio.sleep(1)
            if not self.process:
                return

            error_info = self.process.get_last_error()
            if error_info:
                error_msg, error_time = error_info
                self.status.last_error = error_msg
                self.status.last_error_time = error_time
                await self._emit_monitoring_event({
                    "type": "error",
                    "pipeline": self.pipeline,
                    "error": error_msg,
                    "time": error_time
                })

            current_time = time.time()
            last_input_time = self.status.last_input_time or start_time
            last_output_time = self.status.last_output_time or start_time
            last_params_update_time = self.status.last_params_update_time or start_time

            time_since_last_input = current_time - last_input_time
            time_since_last_output = current_time - last_output_time
            time_since_start = current_time - start_time
            time_since_last_params = current_time - last_params_update_time
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
                time_since_last_output > 8
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
        start_time = 0.0
        try:
            async for frame in self.protocol.ingress_loop(done):
                if done.is_set() or not self.process:
                    return
                if not start_time:
                    start_time = time.time()

                # crop the max square from the center of the image and scale to 512x512
                # most models expect this size especially when using tensorrt
                if frame.size != (512, 512):
                    frame_array = np.array(frame)
                    height, width = frame_array.shape[:2]

                    if width != height:
                        square_size = min(width, height)
                        start_x = width // 2 - square_size // 2
                        start_y = height // 2 - square_size // 2
                        frame_array = frame_array[start_y:start_y+square_size, start_x:start_x+square_size]

                    # Resize using cv2 (much faster than PIL)
                    frame_array = cv2.resize(frame_array, (512, 512))
                    frame = Image.fromarray(frame_array)

                logging.debug(f"Sending input frame. Scaled from {width}x{height} to {frame.size[0]}x{frame.size[1]}")
                self.process.send_input(frame)
                self.status.last_input_time = time.time()  # Track time after send completes

                # Increment frame count and measure FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= fps_log_interval:
                    self.status.input_fps = frame_count / elapsed_time
                    logging.info(f"Input FPS: {self.status.input_fps:.2f}")
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
            start_time = 0.0
            while not done.is_set() and self.process:
                output_image = await self.process.recv_output()
                if not start_time:
                    # only start measuring output FPS after the first frame
                    start_time = time.time()
                if not output_image:
                    break

                self.status.last_output_time = time.time()  # Track time after receive completes
                logging.debug(
                    f"Output image received out_width: {output_image.width}, out_height: {output_image.height}"
                )

                yield output_image

                # Increment frame count and measure FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= fps_log_interval:
                    self.status.output_fps = frame_count / elapsed_time
                    logging.info(f"Output FPS: {self.status.output_fps:.2f}")
                    frame_count = 0
                    start_time = time.time()

        try:
            await self.protocol.egress_loop(gen_output_frames())
            # automatically stop the streamer when the egress ends cleanly
            await self.stop()
        except Exception:
            logging.error("Error running egress loop", exc_info=True)
            await self._restart()

    async def run_control_loop(self):
        """Consumes control messages from the protocol and updates parameters"""
        try:
            async for params in self.protocol.control_loop():
                try:
                    await self.update_params(params)
                except Exception as e:
                    logging.error(f"Error updating model with control message: {e}")
            logging.info("Control loop ended")
            # control loop it not required to be running, so we keep the streamer running
        except Exception as e:
            logging.error(f"Error in control loop", exc_info=True)
