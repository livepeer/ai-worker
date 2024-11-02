import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError

logger = logging.getLogger(__name__)


class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = get_model_dir()
        self.torch_device = get_torch_device()
        self.infer_script_path = (
            Path(__file__).parent.parent / "live" / "infer.py"
        )
        self.process = None
        self.monitor_thread = None
        self.log_thread = None

    def __call__(
        self, **kwargs
    ):
        try:
            if not self.process:
                self.start_process(
                    pipeline=self.model_id,  # we use the model_id as the pipeline name for now
                    input_address="tcp://localhost:5555",
                    output_address="tcp://localhost:5556",
                    http_port=8888,
                    subscribe_url=kwargs["subscribe_url"],
                    publish_url=kwargs["publish_url"],
                    initial_params=json.dumps(kwargs["params"]),
                    # TODO: set torch device from self.torch_device
                )
            logger.info(f"Starting stream, subscribe={kwargs['subscribe_url']} publish={kwargs['publish_url']}")
            return
        except Exception as e:
            raise InferenceError(original_exception=e)

    def start_process(self, **kwargs):
        cmd = ["python", str(self.infer_script_path)]

        # Add any additional kwargs as command-line arguments
        for key, value in kwargs.items():
            kebab_key = key.replace("_", "-")
            if isinstance(value, str):
                escaped_value = str(value).replace("'", "'\\''")
                cmd.extend([f"--{kebab_key}", f"{escaped_value}"])
            else:
                cmd.extend([f"--{kebab_key}", f"{value}"])

        env = os.environ.copy()
        env["HUGGINGFACE_HUB_CACHE"] = self.model_dir

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )

            self.monitor_thread = threading.Thread(target=self.monitor_process)
            self.monitor_thread.start()
            self.log_thread = threading.Thread(target=log_output, args=(self.process.stdout,))
            self.log_thread.start()

        except subprocess.CalledProcessError as e:
            raise InferenceError(f"Error starting infer.py: {e}")

    def monitor_process(self):
        while True:
            return_code = self.process.poll()
            if return_code is not None:
                logger.info(f"infer.py process completed. Return code: {return_code}")
                if return_code != 0:
                    _, stderr = self.process.communicate()
                    logger.error(
                        f"infer.py process failed with return code {return_code}. Error: {stderr}"
                    )
                break

            logger.info("infer.py process is running...")
            time.sleep(10)

    def stop_process(self):
        if self.process:
            self.process.terminate()
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.log_thread:
            self.log_thread.join()

    def __str__(self) -> str:
        return f"VideoToVideoPipeline model_id={self.model_id}"


def log_output(f):
    for line in f:
        sys.stderr.write(line)
