import asyncio
import logging
import queue
import json
from typing import AsyncGenerator, Optional

from PIL import Image
from multiprocessing.synchronize import Event

from trickle import media, TricklePublisher

from .protocol import StreamProtocol
from .jpeg import to_jpeg_bytes, from_jpeg_bytes

class TrickleProtocol(StreamProtocol):
    def __init__(self, subscribe_url: str, publish_url: str, events_url: Optional[str] = None):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.events_url = events_url
        self.subscribe_queue = queue.Queue[bytearray]()
        self.publish_queue = queue.Queue[bytearray]()
        self.events_publisher = None
        self.subscribe_task = None
        self.publish_task = None

    async def start(self):
        self.subscribe_task = asyncio.create_task(
            media.run_subscribe(self.subscribe_url, self.subscribe_queue.put)
        )
        self.publish_task = asyncio.create_task(
            media.run_publish(self.publish_url, self.publish_queue.get)
        )
        if self.events_url:
            self.events_publisher = TricklePublisher(self.events_url, "application/json")

    async def stop(self):
        if not self.subscribe_task or not self.publish_task:
            return

        # send sentinel None values to stop the trickle tasks gracefully
        self.subscribe_queue.put(None)
        self.publish_queue.put(None)

        await self.events_publisher.close()
        self.events_publisher = None

        tasks = [self.subscribe_task, self.publish_task]
        try:
            await asyncio.wait(tasks, timeout=30.0)
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()

        self.subscribe_task = None
        self.publish_task = None

    async def ingress_loop(self, done: Event) -> AsyncGenerator[Image.Image, None]:
        def dequeue_jpeg():
            jpeg_bytes = self.subscribe_queue.get()
            if not jpeg_bytes:
                return None
            try:
                return from_jpeg_bytes(jpeg_bytes)
            except Exception as e:
                logging.error(f"Error decoding JPEG: {e}")
                raise e

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

    async def report_status(self, status: dict):
        if not self.events_publisher:
            return
        try:
            status_json = json.dumps(status)
            async with await self.events_publisher.next() as event:
                event.write(status_json.encode())
        except Exception as e:
            logging.error(f"Error reporting status: {e}")
