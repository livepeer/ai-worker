import argparse
import asyncio
import aiohttp
import sys
import logging

from trickle_reader import TrickleReader

async def main():

    # Handle CLI args
    parser = argparse.ArgumentParser(description="trickle a reader to a pipe")
    parser.add_argument('--url', type=str, default="http://localhost:2939", help="Trickle server to use")
    parser.add_argument('--stream', type=str, required=True, help="Stream name")
    args = parser.parse_args()

    reader = TrickleReader(base_url=args.url, stream_name=args.stream)

    # Simulate reading multiple segments
    for _ in range(75):
        segment = None
        try:
            segment = await reader.next()
            while segment:
                chunk = await segment.read()
                if not chunk:
                    break  # End of segment
                # Write the binary data to stdout (sys.stdout.buffer for binary output)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()  # Ensure data is flushed to stdout
        except aiohttp.ClientError as e:
            logging.error(f"Failed to read the segment content: {e} - read {kount}")
            break # End of stream?
        finally:
            await segment.close()

if __name__ == "__main__":
    asyncio.run(main())

