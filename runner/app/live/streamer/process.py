import os
import asyncio
import logging
import multiprocessing as mp
import queue
import time

from PIL import Image

from pipelines import load_pipeline


class PipelineProcess:
    @staticmethod
    def start(pipeline_name: str, **params):
        instance = PipelineProcess(pipeline_name)
        if params:
            instance.update_params(**params)
        instance.process.start()
        return instance

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.ctx = mp.get_context("spawn")

        self.input_queue = self.ctx.Queue(maxsize=5)
        self.last_input_time = 0.0
        self.output_queue = self.ctx.Queue()
        self.last_output_time = 0.0
        self.param_update_queue = self.ctx.Queue()

        self.done = self.ctx.Event()
        self.process = self.ctx.Process(target=self.process_loop, args=())

    def stop(self):
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

        for q in [self.input_queue, self.output_queue, self.param_update_queue]:
            q.cancel_join_thread()
            q.close()
        self.done = None

    def is_done(self):
        return self.done.is_set()

    def update_params(self, **params):
        self.param_update_queue.put(params)

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

    async def recv_output(self) -> Image.Image | None:
        # we cannot do a long get with timeout as that would block the asyncio
        # event loop, so we loop with nowait and sleep async instead.
        # TODO: use asyncio.to_thread instead
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
        level = logging.DEBUG if os.environ.get('VERBOSE_LOGGING') == '1' else logging.INFO
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=level,
            datefmt='%Y-%m-%d %H:%M:%S')

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
                    continue

                try:
                    output_image = pipeline.process_frame(input_image)
                    self.output_queue.put(output_image)
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
        except Exception as e:
            logging.error(f"Error in process run method: {e}", exc_info=True)
