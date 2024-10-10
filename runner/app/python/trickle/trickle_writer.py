import asyncio
import aiohttp
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)

class TrickleWriter:
    def __init__(self, base_url: str, stream_name: str):
        self.base_url = f"{base_url}/realtime/{stream_name}"
        self.stream_name = stream_name
        self.idx = 0  # Start index for POSTs
        self.next_writer = None
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session = aiohttp.ClientSession()

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()

    def streamIdx(self):
        return f"{self.stream_name}/{self.idx}"

    async def preconnect(self):
        """Preconnect to the server by initiating a POST request to the current index."""
        url = f"{self.base_url}/{self.idx}"
        logging.info(f"Preconnecting to URL: {url}")
        try:
            # we will be incrementally writing data into this queue
            queue = asyncio.Queue()
            asyncio.create_task(self._run_post(url, queue))
            return queue
        except aiohttp.ClientError as e:
            logging.error(f"Failed to complete POST for {self.streamIdx()}: {e}")
            return None

    async def _run_post(self, url, queue):
        resp = await self.session.post(
            url, 
            # TODO content type
            headers={'Connection': 'close'}, 
            data=self._stream_data(queue)
        )
        # TODO propagate errors?
        if resp.status != 200:
            body = await resp.text()
            logging.error(f"Failed POST {self.streamIdx()}, status code: {resp.status}, msg: {body}")
        return None

    async def _stream_data(self, queue):
        """Stream data from the queue for the POST request."""
        while True:
            chunk = await queue.get()
            if chunk is None:  # Stop signal
                break
            yield chunk

    async def next(self):
        """Start or retrieve a pending POST request and preconnect for the next segment."""
        async with self.lock:
            if self.next_writer is None:
                logging.info(f"No pending connection, preconnecting {self.streamIdx()}...")
                self.next_writer = await self.preconnect()

            writer = self.next_writer
            self.next_writer = None

            # Set up the next POST in the background
            asyncio.create_task(self._preconnect_next_segment())

        return SegmentWriter(writer)

    async def _preconnect_next_segment(self):
        """Preconnect to the next POST in the background."""
        logging.info(f"Setting up next connection for {self.streamIdx()}")
        async with self.lock:
            if self.next_writer is not None:
                return
            self.idx += 1  # Increment the index for the next POST
            next_writer = await self.preconnect()
            if next_writer:
                self.next_writer = next_writer

    def _data_generator(self):
        """Placeholder for a real data generator for chunked transfer."""
        while True:
            yield b"chunk"

    async def close(self):
        """Close the session when done."""
        logging.info(f"Closing {self.base_url}")
        if self.next_writer:
            s = SegmentWriter(self.next_writer)
            await s.close()
        await self.session.delete(self.base_url)
        await self.session.close()

class SegmentWriter:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def write(self, data):
        """Write data to the current segment."""
        await self.queue.put(data)

    async def close(self):
        """Ensure the request is properly closed when done."""
        await self.queue.put(None)  # Send None to signal end of data

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the connection."""
        await self.close()
