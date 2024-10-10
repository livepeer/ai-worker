import argparse
import asyncio
import aiohttp
import logging
import os

from trickle_writer import TrickleWriter

async def main():

    # Handle CLI args
    parser = argparse.ArgumentParser(description="Trickle Writer")
    parser.add_argument("--url", default="http://localhost:2939", help="Base URL (default: http://localhost:2939)")
    parser.add_argument("--stream", type=str, required=True, help="Stream name (required)")
    parser.add_argument("--local", type=str, required=True, help="Path to local text file (required)")
    args = parser.parse_args()

    # Validate the local file path
    if not os.path.isfile(args.local):
        logging.error(f"The file '{args.local}' does not exist.")
        return

    # Initialize the TrickleWriter
    writer = TrickleWriter(args.url, args.stream)

    kount = 0

    try:
        # Open the local file and read it line by line
        with open(args.local, 'r') as f:
            while True:
                # Start a new segment every 20 lines
                async with await writer.next() as segment:
                    # Write up to 20 lines per segment
                    for _ in range(20):
                        line = await _async_line_reader(f).__anext__()  # Read the next line
                        await segment.write(line.encode())  # Write the line as bytes
                        await asyncio.sleep(0.1)  # Pause for 100ms between lines
                kount+=1
                if kount > 1000:
                    break
    finally:
        await writer.close()


async def _async_line_reader(file):
    """Async generator to read lines from a file."""
    loop = asyncio.get_event_loop()
    for line in file:
        yield line


if __name__ == "__main__":
    asyncio.run(main())
