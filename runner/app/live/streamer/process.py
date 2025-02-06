import os
import asyncio
import logging
import multiprocessing as mp
import queue
import sys
import time
from typing import Any

from PIL import Image

from pipelines import load_pipeline

from trickle import InputFrame, AudioFrame, VideoFrame, OutputFrame, VideoOutput, AudioOutput

class PipelineProcess:
    @staticmethod
    def start(pipeline_name: str, params: dict, request_id: str):
        instance = PipelineProcess(pipeline_name, request_id)
        if params:
            instance.update_params(params)
        instance.process.start()
        instance.start_time = time.time()
        return instance

    def __init__(self, pipeline_name: str, request_id: str):
        self.pipeline_name = pipeline_name
        self.ctx = mp.get_context("spawn")

        self.input_queue = self.ctx.Queue(maxsize=5)
        self.output_queue = self.ctx.Queue()
        self.param_update_queue = self.ctx.Queue()
        self.error_queue = self.ctx.Queue()
        self.log_queue = self.ctx.Queue(maxsize=100)  # Keep last 100 log lines

        self.done = self.ctx.Event()
        self.process = self.ctx.Process(target=self.process_loop, args=())
        self.start_time = 0
        self.request_id = request_id

    async def stop(self):
        await asyncio.to_thread(self._stop_sync)

    def _stop_sync(self):
        self.done.set()
        if not self.process.is_alive():
            logging.info("Process already not alive")
            return

        logging.info("Terminating pipeline process")
        self.process.terminate()

        stopped = False
        try:
            self.process.join(timeout=5)
            stopped = True
        except Exception as e:
            logging.error(f"Process join error: {e}")
        if not stopped or self.process.is_alive():
            logging.error("Failed to terminate process, killing")
            self.process.kill()

        for q in [self.input_queue, self.output_queue, self.param_update_queue,
                  self.error_queue, self.log_queue]:
            q.cancel_join_thread()
            q.close()

    def is_done(self):
        return self.done.is_set()

    def update_params(self, params: dict):
        self.param_update_queue.put(params)

    def send_input(self, frame: InputFrame):
        self._queue_put_fifo(self.input_queue, frame)

    async def recv_output(self) -> OutputFrame | None:
        # we cannot do a long get with timeout as that would block the asyncio
        # event loop, so we loop with nowait and sleep async instead.
        # TODO: use asyncio.to_thread instead
        while not self.is_done():
            try:
                output = self.output_queue.get_nowait()
                return output
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue
        return None

    def get_recent_logs(self, n=10) -> list[str]:
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs[-n:]  # Return last n logs

    def process_loop(self):
        self._setup_logging()

        def report_error(error_msg: str):
            error_event = {
                "message": error_msg,
                "timestamp": time.time()
            }
            logging.error(error_msg)
            self._queue_put_fifo(self.error_queue, error_event)

        try:
            params = {}
            try:
                params = self.param_update_queue.get_nowait()
            except queue.Empty:
                pass
            except Exception as e:
                report_error(f"Error getting params: {e}")

            try:
                pipeline = load_pipeline(self.pipeline_name, **params)
                logging.info("Pipeline loaded successfully")
            except Exception as e:
                report_error(f"Error loading pipeline: {e}")
                try:
                    pipeline = load_pipeline(self.pipeline_name)
                    logging.info("Pipeline loaded with default params")
                except Exception as e:
                    report_error(f"Error loading pipeline with default params: {e}")
                    raise

            while not self.is_done():
                if not self.param_update_queue.empty():
                    params = self.param_update_queue.get_nowait()
                    try:
                        pipeline.update_params(**params)
                        logging.info(f"Updated params: {params}")
                    except Exception as e:
                        report_error(f"Error updating params: {e}")

                try:
                    input_frame = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    if isinstance(input_frame, VideoFrame):
                        output_image = pipeline.process_frame(input_frame.image)
                        output_frame = VideoOutput(input_frame.replace_image(output_image))
                        self.output_queue.put(output_frame)
                    elif isinstance(input_frame, AudioFrame):
                        self.output_queue.put(AudioOutput([input_frame]))
                        # TODO wire in a proper pipeline here
                    else:
                        report_error(f"Unsupported input frame type {type(input_frame)}")
                except Exception as e:
                    report_error(f"Error processing frame: {e}")
        except Exception as e:
            report_error(f"Error in process run method: {e}")

    def _setup_logging(self):

        level = logging.DEBUG if os.environ.get('VERBOSE_LOGGING') == '1' else logging.INFO
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s request_id=' + self.request_id + ' %(message)s',
            level=level,
            datefmt='%Y-%m-%d %H:%M:%S')

        queue_handler = LogQueueHandler(self)
        queue_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s request_id=' + self.request_id + ' %(message)s'))
        logging.getLogger().addHandler(queue_handler)

        # Tee stdout and stderr to our log queue while preserving original output
        sys.stdout = QueueTeeStream(sys.stdout, self)
        sys.stderr = QueueTeeStream(sys.stderr, self)

    def _queue_put_fifo(self, _queue: mp.Queue, item: Any):
        """Helper to put an item on a queue, dropping oldest items if needed"""
        while not self.is_done():
            try:
                _queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    _queue.get_nowait()  # remove oldest item
                except queue.Empty:
                    continue

    def get_last_error(self) -> tuple[str, float] | None:
        """Get the most recent error and its timestamp from the error queue, if any"""
        last_error = None
        while True:
            try:
                last_error = self.error_queue.get_nowait()
            except queue.Empty:
                break
        return (last_error["message"], last_error["timestamp"]) if last_error else None

class QueueTeeStream:
    """Tee all stream (stdout or stderr) messages to the process log queue"""
    def __init__(self, original_stream, process: PipelineProcess):
        self.original_stream = original_stream
        self.process = process

    def write(self, text):
        self.original_stream.write(text)
        text = text.strip()  # Only queue non-empty lines
        if text:
            self.process._queue_put_fifo(self.process.log_queue, text)

    def flush(self):
        self.original_stream.flush()

class LogQueueHandler(logging.Handler):
    """Send all log records to the process's log queue"""
    def __init__(self, process: PipelineProcess):
        super().__init__()
        self.process = process

    def emit(self, record):
        msg = self.format(record)
        self.process._queue_put_fifo(self.process.log_queue, msg)
