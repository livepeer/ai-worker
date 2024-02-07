import argparse
from time import time
from typing import List
import torch
from PIL import Image
from app.main import load_pipeline
from app.pipelines import (
    Pipeline,
    TextToImagePipeline,
    ImageToImagePipeline,
    ImageToVideoPipeline,
)

PROMPT = "a mountain lion"
IMAGE = "images/test.png"


def call_pipeline(pipeline: Pipeline, batch_size=1) -> List[any]:
    if isinstance(pipeline, TextToImagePipeline):
        prompts = [PROMPT] * batch_size
        return pipeline(prompts)
    elif isinstance(pipeline, ImageToImagePipeline):
        prompts = [PROMPT] * batch_size
        images = [Image.open(IMAGE).convert("RGB")] * batch_size
        return pipeline(prompts, images)
    elif isinstance(pipeline, ImageToVideoPipeline):
        images = [Image.open(IMAGE).convert("RGB")] * batch_size
        return pipeline(images)
    else:
        raise Exception("invalid pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A benchmarking tool for AI pipelines")
    parser.add_argument(
        "--pipeline", type=str, required=True, help="the name of the pipeline"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="the ID of the model to load for the pipeline",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        required=False,
        help="the number of times to call the pipeline",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, required=False, help="the size of a batch"
    )

    args = parser.parse_args()

    start = time()
    config = load_pipeline(args.pipeline, args.model_id)
    pipeline = config["pipeline"]

    max_mem_allocated = torch.cuda.max_memory_allocated()
    max_mem_reserved = torch.cuda.max_memory_reserved()

    print(f"pipeline load time: {time() - start:.3f}s")
    print(
        f"pipeline load max GPU memory allocated: {max_mem_allocated / 1024**3:.3f}GiB"
    )
    print(f"pipeline load max GPU memory reserved: {max_mem_reserved / 1024**3:.3f}GiB")

    for i in range(args.runs):
        batch_size = args.batch_size

        start = time()
        output = call_pipeline(pipeline, batch_size)
        assert len(output) == batch_size

        inference_time = time() - start
        max_mem_allocated = torch.cuda.max_memory_allocated()
        max_mem_reserved = torch.cuda.max_memory_reserved()

        print(f"inference {i} {batch_size=} time: {inference_time:.3f}s")
        print(
            f"inference {i} {batch_size=} time per output: {inference_time / batch_size:.3f}s"
        )
        print(
            f"inference {i} {batch_size=} max GPU memory allocated: {max_mem_allocated / 1024**3:.3f}GiB"
        )
        print(
            f"inference {i} {batch_size=} max GPU memory reserved: {max_mem_reserved / 1024**3:.3f}GiB"
        )
