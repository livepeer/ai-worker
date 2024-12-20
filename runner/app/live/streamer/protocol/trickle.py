import asyncio
import logging
import queue
import json
from typing import AsyncGenerator, Optional

from PIL import Image

from trickle import media, TricklePublisher, TrickleSubscriber

from .protocol import StreamProtocol
from .jpeg import to_jpeg_bytes, from_jpeg_bytes

class TrickleProtocol(StreamProtocol):
    def __init__(self, subscribe_url: str, publish_url: str, control_url: Optional[str] = None, events_url: Optional[str] = None):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.subscribe_queue = queue.Queue[bytes | None]()
        self.publish_queue = queue.Queue[bytes | None]()
        self.control_subscriber = None
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
        if self.control_url and self.control_url.strip() != "":
            self.control_subscriber = TrickleSubscriber(self.control_url)
        if self.events_url and self.events_url.strip() != "":
            self.events_publisher = TricklePublisher(self.events_url, "application/json")

    async def stop(self):
        if not self.subscribe_task or not self.publish_task:
            return # already stopped

        # send sentinel None values to stop the trickle tasks gracefully
        cancel_tasks = [
            asyncio.to_thread(self.subscribe_queue.put, None),
            asyncio.to_thread(self.publish_queue.put, None)
        ]
        if self.control_subscriber:
            cancel_tasks.append(self.control_subscriber.close())
        if self.events_publisher:
            cancel_tasks.append(self.events_publisher.close())
        # now schedule all of them to run in background
        cancel_tasks = [asyncio.create_task(t) for t in cancel_tasks]

        try:
            work_tasks = [self.subscribe_task, self.publish_task]
            _, pending = await asyncio.wait(cancel_tasks + work_tasks, timeout=10)
            # cancel all the pending work tasks, let the cancel tasks finish in background
            for task in pending:
                if task in work_tasks:
                    task.cancel()
        finally:
            self.subscribe_task = None
            self.publish_task = None
            self.control_subscriber = None
            self.events_publisher = None

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[Image.Image, None]:
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

    async def emit_monitoring_event(self, event: dict):
        if not self.events_publisher:
            return
        try:
            event_json = json.dumps(event)
            async with await self.events_publisher.next() as writer:
                await writer.write(event_json.encode())
        except Exception as e:
            logging.error(f"Error reporting status: {e}")

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        if not self.control_subscriber:
            logging.warning("No control-url provided, inference won't get updates from the control trickle subscription")
            return

        logging.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        while not done.is_set():
            try:
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    return

                params = ''
                while True:
                    chunk = await segment.read()
                    if not chunk:
                        break
                    params += chunk.decode()
                data = json.loads(params)
                if data == keepalive_message:
                    # Ignore periodic keepalive messages
                    continue

                logging.info("Received control message with params: %s", data)
                yield data

            except Exception:
                logging.error(f"Error in control loop", exc_info=True)
                continue

