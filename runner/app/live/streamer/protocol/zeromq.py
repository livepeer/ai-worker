import io

import zmq.asyncio
from PIL import Image
from multiprocessing.synchronize import Event
from typing import AsyncGenerator, Optional

from .protocol import StreamProtocol
from .jpeg import to_jpeg_bytes, from_jpeg_bytes


class ZeroMQProtocol(StreamProtocol):
    def __init__(self, input_address: str, output_address: str):
        self.input_address = input_address
        self.output_address = output_address
        self.context = zmq.asyncio.Context()
        self.input_socket = self.context.socket(zmq.SUB)
        self.output_socket = self.context.socket(zmq.PUB)

    async def start(self):
        self.input_socket.connect(self.input_address)
        self.input_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.input_socket.set_hwm(10)
        self.output_socket.connect(self.output_address)
        self.output_socket.set_hwm(10)

    async def stop(self):
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

    async def report_status(self, status: dict):
        pass  # No-op for ZeroMQ

    async def control_loop(self) -> AsyncGenerator[dict, None]:
        if False:
            yield {}  # Empty generator, dummy yield for proper typing
        pass # ZeroMQ protocol does not support control messages
