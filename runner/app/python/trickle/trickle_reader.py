import asyncio
import aiohttp
import logging
import sys

logging.basicConfig(level=logging.INFO)

class TrickleReader:
    def __init__(self, base_url: str, stream_name: str):
        self.base_url = f"{base_url}/realtime/{stream_name}"
        self.stream_name = stream_name
        self.idx = -1  # Start with -1 for 'latest' index
        self.pending_get = None  # Pre-initialized GET request
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session = aiohttp.ClientSession()

    async def get_index(self, resp):
        """Extract the index from the response headers."""
        if resp is None:
            return -1
        idx_str = resp.headers.get("Lp-Trickle-Idx")
        try:
            idx = int(idx_str)
        except (TypeError, ValueError):
            return -1
        return idx

    async def preconnect(self):
        """Preconnect to the server by making a GET request to fetch the next segment."""
        url = f"{self.base_url}/{self.idx}"
        logging.info(f"Preconnecting to URL: {url}")
        try:

            resp = await self.session.get(url, headers={'Connection':'close'})
            if resp.status != 200:
                body = await resp.text()
                resp.release()
                logging.error(f"Failed GET segment, status code: {resp.status}, msg: {body}")
                return None

            # Return the response for later processing
            return resp
        except aiohttp.ClientError as e:
            logging.error(f"Failed to complete GET for next segment: {e}")
            return None

    async def next(self):
        """Retrieve data from the current segment and set up the next segment concurrently."""
        async with self.lock:
            # If we don't have a pending GET request, preconnect
            if self.pending_get is None:
                logging.info("No pending connection, preconnecting...")
                self.pending_get = await self.preconnect()

            # Extract the current connection to use for reading
            conn = self.pending_get
            self.pending_get = None

            # Extract and set the next index from the response headers
            idx = await self.get_index(conn)
            if idx != -1:
                self.idx = idx + 1

            # Set up the next connection in the background
            asyncio.create_task(self._preconnect_next_segment())

        return Segment(conn)

    async def _preconnect_next_segment(self):
        """Preconnect to the next segment in the background."""
        logging.info(f"Setting up next connection for index {self.idx}")
        async with self.lock:
            if self.pending_get is not None:
                return
            next_conn = await self.preconnect()
            if next_conn:
                self.pending_get = next_conn
                next_idx = await self.get_index(next_conn)
                if next_idx != -1:
                    self.idx = next_idx + 1

class Segment:
    def __init__(self, response):
        self.response = response

    async def read(self, chunk_size=2048):
        """Read the next chunk of the segment."""
        chunk = await self.response.content.read(chunk_size)
        if not chunk:
            await self.close()
        return chunk

    async def close(self):
        """Ensure the response is properly closed when done."""
        if not self.response.closed:
            await self.response.release()
            await self.response.close()
