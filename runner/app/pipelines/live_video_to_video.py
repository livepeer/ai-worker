import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
import time
import asyncio
from typing import IO
from pydantic import BaseModel
import http.client

from app.pipelines.base import Pipeline, HealthCheck
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError

from app.live.streamer import ProcessGuardian, PipelineStreamer

class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = get_model_dir()
        self.torch_device = get_torch_device()
        self.infer_script_path = (
            Path(__file__).parent.parent / "live" / "infer.py"
        )

        try:
            logging.info("Starting pipeline process")
            self.process = ProcessGuardian(self.model_id, {})
            self.streamer: PipelineStreamer | None = None
        except Exception as e:
            raise InferenceError(original_exception=e)


    def __call__(  # type: ignore
        self, *, subscribe_url: str, publish_url: str, control_url: str, events_url: str, params: dict, request_id: str, stream_id: str, **kwargs
    ):
        if not self.process:
            raise RuntimeError("Pipeline process not running")

        max_retries = 10
        thrown_ex = None
        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(1)
            try:
                if self.streamer:
                    asyncio.run(self.streamer.stop(timeout=10))

                conn = http.client.HTTPConnection("localhost", 8888)
                conn.request(
                    "POST",
                    "/api/live-video-to-video",
                    json.dumps(
                        {
                            "subscribe_url": subscribe_url,
                            "publish_url": publish_url,
                            "control_url": control_url,
                            "events_url": events_url,
                            "params": params,
                            "request_id": request_id or "",
                            "stream_id": stream_id or "",
                        }
                    ),
                    headers={"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                if response.status != 200:
                    continue

                logging.info("Stream started successfully")
                return {} # Return an empty success response
            except Exception as e:
                thrown_ex = e
                logging.error(f"Attempt {attempt + 1} failed", exc_info=True)

        raise InferenceError(original_exception=thrown_ex)

    def get_health(self) -> HealthCheck:
        if not self.process:
            # The infer process is supposed to be always running, so if it's
            # gone it means an ERROR and the worker is allowed to kill us.
            return HealthCheck(status="ERROR")

        try:
            pipe_status = self.process.get_status()
            return HealthCheck(
                status="IDLE" if pipe_status.state == "OFFLINE" else "OK"
            )
        except Exception as e:
            logging.error(f"Failed to get status", exc_info=True)
            raise ConnectionError(f"Failed to get status: {e}")

    def __str__(self) -> str:
        return f"VideoToVideoPipeline model_id={self.model_id}"


def log_output(f: IO[str]):
    try:
        for line in f:
            sys.stderr.write(line)
    except Exception as e:
        logging.error(f"Error while logging process output: {e}")
