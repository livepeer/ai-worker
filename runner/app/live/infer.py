import argparse
import asyncio
import json
import logging
import signal
import sys
import os
import traceback
from typing import List
import logging

from streamer import PipelineStreamer

# loads neighbouring modules with absolute paths
infer_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, infer_root)

from api import start_http_server
from streamer.protocol.trickle import TrickleProtocol
from streamer.protocol.zeromq import ZeroMQProtocol


async def main(*, http_port: int, stream_protocol: str, subscribe_url: str, publish_url: str, control_url: str, events_url: str, pipeline: str, params: dict, input_timeout: int, request_id: str):
    if stream_protocol == "trickle":
        protocol = TrickleProtocol(subscribe_url, publish_url, control_url, events_url)
    elif stream_protocol == "zeromq":
        if events_url:
            logging.warning("ZeroMQ protocol does not support event streaming")
        if control_url:
            logging.warning("ZeroMQ protocol does not support control messages")
        protocol = ZeroMQProtocol(subscribe_url, publish_url)
    else:
        raise ValueError(f"Unsupported protocol: {stream_protocol}")

    streamer = PipelineStreamer(protocol, pipeline, input_timeout, params or {}, request_id)

    api = None
    try:
        await streamer.start()
        api = await start_http_server(http_port, streamer)

        tasks: List[asyncio.Task] = []
        tasks.append(asyncio.create_task(streamer.wait()))
        tasks.append(asyncio.create_task(block_until_signal([signal.SIGINT, signal.SIGTERM])))

        await asyncio.wait(tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
    except Exception as e:
        logging.error(f"Error starting socket handler or HTTP server: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e
    finally:
        await streamer.stop(timeout=5)
        await api.cleanup()


async def block_until_signal(sigs: List[signal.Signals]):
    loop = asyncio.get_running_loop()
    future: asyncio.Future[signal.Signals] = loop.create_future()

    def signal_handler(sig, _):
        logging.info(f"Received signal: {sig}")
        loop.call_soon_threadsafe(future.set_result, sig)

    for sig in sigs:
        signal.signal(sig, signal_handler)
    return await future

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer process to run the AI pipeline")
    parser.add_argument(
        "--http-port", type=int, default=8888, help="Port for the HTTP server"
    )
    parser.add_argument(
        "--pipeline", type=str, default="streamdiffusion", help="Pipeline to use"
    )
    parser.add_argument(
        "--initial-params", type=str, default="{}", help="Initial parameters for the pipeline"
    )
    parser.add_argument(
        "--stream-protocol",
        type=str,
        choices=["trickle", "zeromq"],
        default=os.getenv("STREAM_PROTOCOL", "trickle"),
        help="Protocol to use for streaming frames in and out. One of: trickle, zeromq"
    )
    parser.add_argument(
        "--subscribe-url", type=str, required=True, help="URL to subscribe for the input frames (trickle). For zeromq this is the input socket address"
    )
    parser.add_argument(
        "--publish-url", type=str, required=True, help="URL to publish output frames (trickle). For zeromq this is the output socket address"
    )
    parser.add_argument(
        "--control-url", type=str, help="URL to subscribe for Control API JSON messages to update inference params"
    )
    parser.add_argument(
        "--events-url", type=str, help="URL to publish events about pipeline status and logs."
    )
    parser.add_argument(
        "--input-timeout",
        type=int,
        default=60,
        help="Timeout in seconds to wait after input frames stop before shutting down. Set to 0 to disable."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default="",
        help="The Livepeer request ID associated with this video stream"
    )
    args = parser.parse_args()
    try:
        params = json.loads(args.initial_params)
    except Exception as e:
        logging.error(f"Error parsing --initial-params: {e}")
        sys.exit(1)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s request_id=' + args.request_id + ' %(message)s',
        level=log_level,
        datefmt='%Y-%m-%d %H:%M:%S')
    if args.verbose:
        os.environ['VERBOSE_LOGGING'] = '1' # enable verbose logging in subprocesses

    # formatter = logging.Formatter(
    #     '%(asctime)s %(levelname)-8s request_id=%(request_id)s %(message)s',
    #     defaults={"request_id": ""},
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )
    # handler = logging.StreamHandler()
    # handler.setFormatter(formatter)
    # handler.setLevel(log_level)
    # logger = logging.getLogger()  # Root logger
    # logger.setLevel(log_level)
    # logger.addHandler(handler)
    # logger.addFilter(ContextFilter())  # Attach the filter to all loggers
    # logging_config.request_id = args.request_id

    try:
        asyncio.run(
            main(
                http_port=args.http_port,
                stream_protocol=args.stream_protocol,
                subscribe_url=args.subscribe_url,
                publish_url=args.publish_url,
                control_url=args.control_url,
                events_url=args.events_url,
                pipeline=args.pipeline,
                params=params,
                input_timeout=args.input_timeout,
                request_id=args.request_id
            )
        )
        # We force an exit here to ensure that the process terminates. If any asyncio tasks or
        # sub-processes failed to shutdown they'd block the main process from exiting.
        os._exit(0)
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        logging.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        os._exit(1)

