import io

import zmq.asyncio
from PIL import Image
from multiprocessing.synchronize import Event
from typing import AsyncGenerator

from .streamer import PipelineStreamer


def to_jpeg_bytes(frame: Image.Image):
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG")
    bytes = buffer.getvalue()
    buffer.close()
    return bytes


def from_jpeg_bytes(frame_bytes: bytes):
    image = Image.open(io.BytesIO(frame_bytes))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


class ZeroMQStreamer(PipelineStreamer):
    def __init__(
        self,
        input_address: str,
        output_address: str,
        pipeline: str,
    ):
        super().__init__(pipeline)
        self.input_address = input_address
        self.output_address = output_address

        self.context = zmq.asyncio.Context()
        self.input_socket = self.context.socket(zmq.SUB)
        self.output_socket = self.context.socket(zmq.PUB)

    def start(self):
        self.input_socket.connect(self.input_address)
        self.input_socket.setsockopt_string(
            zmq.SUBSCRIBE, ""
        )  # Subscribe to all messages
        self.input_socket.set_hwm(10)

        self.output_socket.connect(self.output_address)
        self.output_socket.set_hwm(10)

        super().start()

    async def stop(self):
        await super().stop()
        self.input_socket.close()
        self.output_socket.close()
        self.context.term()

    async def ingress_loop(self, done: Event) -> AsyncGenerator[Image.Image, None]:
        while not done.is_set():
            frame_bytes = await self.input_socket.recv()
            yield from_jpeg_bytes(frame_bytes)

    async def egress_loop(self, output_frames: AsyncGenerator[Image.Image, None]):
        async for frame in output_frames:
            frame_bytes = to_jpeg_bytes(frame)
            await self.output_socket.send(frame_bytes)
