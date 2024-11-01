import asyncio
import logging
import queue

from PIL import Image
from multiprocessing.synchronize import Event
from typing import AsyncGenerator

from trickle import media

from .streamer import PipelineStreamer
from .jpeg import to_jpeg_bytes, from_jpeg_bytes

class TrickleStreamer(PipelineStreamer):
    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        pipeline: str,
        **params,
    ):
        super().__init__(pipeline, **params)
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.subscribe_queue = queue.Queue[bytearray]()
        self.publish_queue = queue.Queue[bytearray]()

    def start(self):
        subscribe_put = lambda m: asyncio.to_thread(self.subscribe_queue.put, m)
        publish_get = lambda: asyncio.to_thread(self.publish_queue.get)

        self.subscribe_task = asyncio.create_task(media.run_subscribe(self.subscribe_url, subscribe_put))
        self.publish_task = asyncio.create_task(media.run_publish(self.publish_url, publish_get))
        super().start()

    async def stop(self):
        if not self.subscribe_task or not self.publish_task:
            await super().stop()
            return

        # send sentinel None values to stop the trickle tasks
        self.subscribe_queue.put(None)
        self.publish_queue.put(None)

        try:
            await asyncio.wait([self.subscribe_task, self.publish_task], timeout=15.0)
        except asyncio.TimeoutError:
            self.subscribe_task.cancel()
            self.publish_task.cancel()
        except Exception:
            logging.error("Error stopping trickle streamer", exc_info=True)

        self.subscribe_task = None
        self.publish_task = None

        await super().stop()

    async def ingress_loop(self, done: Event) -> AsyncGenerator[Image.Image, None]:
        def dequeue_jpeg():
            jpeg_bytes = self.subscribe_queue.get()
            if not jpeg_bytes:
                return None
            return from_jpeg_bytes(jpeg_bytes)

        while not done.is_set():
            image = await asyncio.to_thread(dequeue_jpeg)
            if not image:
                break
            yield image

    async def egress_loop(self, output_frames: AsyncGenerator[Image.Image, None]):
        def enqueue_bytes(frame: Image.Image):
            jpeg_bytes = to_jpeg_bytes(frame)
            self.publish_queue.put(jpeg_bytes)

        async for frame in output_frames:
            await asyncio.to_thread(enqueue_bytes, frame)
