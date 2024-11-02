import asyncio
import aiohttp
import logging
import sys

class TrickleSubscriber:
    def __init__(self, url: str):
        self.base_url = url
        self.idx = -1  # Start with -1 for 'latest' index
        self.pending_get = None  # Pre-initialized GET request
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session = aiohttp.ClientSession()
        self.errored = False

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
        logging.info(f"Trickle sub Preconnecting to URL: {url}")
        try:

            resp = await self.session.get(url, headers={'Connection':'close'})
            if resp.status != 200:
                body = await resp.text()
                resp.release()
                logging.error(f"Trickle sub Failed GET segment, status code: {resp.status}, msg: {body}")
                self.errored = True
                return None

            # Return the response for later processing
            return resp
        except aiohttp.ClientError as e:
            logging.error(f"Trickle sub Failed to complete GET for next segment: {e}")
            self.errored = True
            return None

    async def next(self):
        """Retrieve data from the current segment and set up the next segment concurrently."""
        async with self.lock:

            if self.errored:
                logging.info("Trickle subscription closed or errored")
                return None

            # If we don't have a pending GET request, preconnect
            if self.pending_get is None:
                logging.info("Trickle sub No pending connection, preconnecting...")
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
        logging.info(f"Trickle sub setting up next connection for index {self.idx}")
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

    async def read(self, chunk_size=32 * 1024):
        """Read the next chunk of the segment."""
        if not self.response:
            await self.close()
            return None
        chunk = await self.response.content.read(chunk_size)
        if not chunk:
            await self.close()
        return chunk

    async def close(self):
        """Ensure the response is properly closed when done."""
        if self.response is None:
            return
        if not self.response.closed:
            await self.response.release()
            await self.response.close()
