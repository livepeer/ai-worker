import asyncio
import logging
import queue
import json
from typing import AsyncGenerator, Optional

from PIL import Image

from trickle import media, TricklePublisher, TrickleSubscriber, InputFrame, OutputFrame

from .protocol import StreamProtocol

class TrickleProtocol(StreamProtocol):
    def __init__(self, subscribe_url: str, publish_url: str, control_url: Optional[str] = None, events_url: Optional[str] = None):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.subscribe_queue = queue.Queue[InputFrame]()
        self.publish_queue = queue.Queue[OutputFrame]()
        self.control_subscriber = None
        self.events_publisher = None
        self.subscribe_task = None
        self.publish_task = None

    async def start(self):
        metadata_queue = queue.Queue[dict]() # to pass video metadata from decoder to encoder
        self.subscribe_task = asyncio.create_task(
            media.run_subscribe(self.subscribe_url, self.subscribe_queue.put, metadata_queue.put)
        )
        self.publish_task = asyncio.create_task(
            media.run_publish(self.publish_url, self.publish_queue.get, metadata_queue.get)
        )
        if self.control_url and self.control_url.strip() != "":
            self.control_subscriber = TrickleSubscriber(self.control_url)
        if self.events_url and self.events_url.strip() != "":
            self.events_publisher = TricklePublisher(self.events_url, "application/json")

    async def stop(self):
        if not self.subscribe_task or not self.publish_task:
            return # already stopped

        # send sentinel None values to stop the trickle tasks gracefully
        self.subscribe_queue.put(None)
        self.publish_queue.put(None)

        if self.control_subscriber:
            await self.control_subscriber.close()
            self.control_subscriber = None

        if self.events_publisher:
            await self.events_publisher.close()
            self.events_publisher = None

        tasks = [self.subscribe_task, self.publish_task]
        try:
            await asyncio.wait(tasks, timeout=10.0)
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()

        self.subscribe_task = None
        self.publish_task = None

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        def dequeue_frame():
            frame = self.subscribe_queue.get()
            if not frame:
                return None

            return frame

        while not done.is_set():
            image = await asyncio.to_thread(dequeue_frame)
            if not image:
                break
            yield image

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        def enqueue_bytes(frame: OutputFrame):
            if frame:
                self.publish_queue.put(frame)
            else:
                self.publish_queue.put(None)

        async for frame in output_frames:
            await asyncio.to_thread(enqueue_bytes, frame)

    async def emit_monitoring_event(self, event: dict):
        if not self.events_publisher:
            return
        try:
            event_json = json.dumps(event)
            async with await self.events_publisher.next() as event:
                await event.write(event_json.encode())
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

                params = await segment.read()
                data = json.loads(params)
                if data == keepalive_message:
                    # Ignore periodic keepalive messages
                    continue

                logging.info("Received control message with params: %s", data)
                yield data

            except Exception:
                logging.error(f"Error in control loop", exc_info=True)
                continue

