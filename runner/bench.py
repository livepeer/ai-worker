import argparse
import os
from time import time
from typing import List

import numpy as np
import torch
from app.main import load_pipeline
from app.pipelines.base import Pipeline
from app.pipelines.image_to_image import ImageToImagePipeline
from app.pipelines.image_to_video import ImageToVideoPipeline
from app.pipelines.text_to_image import TextToImagePipeline
from PIL import Image
from pydantic import BaseModel

PROMPT = "a mountain lion"
IMAGE = "images/test.png"


class BenchMetrics(BaseModel):
    inference_time: float
    inference_time_per_output: float
    max_mem_allocated: float
    max_mem_reserved: float


def call_pipeline(pipeline: Pipeline, batch_size=1, **kwargs) -> List[any]:
    if isinstance(pipeline, TextToImagePipeline):
        prompts = [PROMPT] * batch_size
        return pipeline(prompts, **kwargs)
    elif isinstance(pipeline, ImageToImagePipeline):
        prompts = [PROMPT] * batch_size
        images = [Image.open(IMAGE).convert("RGB")] * batch_size
        return pipeline(prompts, images, **kwargs)
    elif isinstance(pipeline, ImageToVideoPipeline):
        images = [Image.open(IMAGE).convert("RGB")] * batch_size
        return pipeline(images, **kwargs)
    else:
        raise Exception("invalid pipeline")


def bench_pipeline(
    pipeline: Pipeline, batch_size=1, runs=1, num_inference_steps=None
) -> BenchMetrics:
    inference_time = np.zeros(runs)
    inference_time_per_output = np.zeros(runs)
    max_mem_allocated = np.zeros(runs)
    max_mem_reserved = np.zeros(runs)

    kwargs = (
        {"num_inference_steps": num_inference_steps}
        if num_inference_steps is not None
        else {}
    )

    for i in range(runs):
        start = time()
        output = call_pipeline(pipeline, batch_size, **kwargs)
        if isinstance(output, tuple):
            output = output[0]
        assert len(output) == batch_size

        inference_time[i] = time() - start
        inference_time_per_output[i] = inference_time[i] / batch_size
        max_mem_allocated[i] = torch.cuda.max_memory_allocated() / 1024**3
        max_mem_reserved[i] = torch.cuda.max_memory_reserved() / 1024**3

        print(f"inference {i} {batch_size=} time: {inference_time[i]:.3f}s")
        print(
            f"inference {i} {batch_size=} time per output: "
            f"{inference_time_per_output[i]:.3f}s"
        )
        print(
            f"inference {i} {batch_size=} max GPU memory allocated: "
            f"{max_mem_allocated[i]:.3f}GiB"
        )
        print(
            f"inference {i} {batch_size=} max GPU memory reserved: "
            f"{max_mem_reserved[i]:.3f}GiB"
        )

    return BenchMetrics(
        inference_time=inference_time.mean(),
        inference_time_per_output=inference_time_per_output.mean(),
        max_mem_allocated=max_mem_allocated.mean(),
        max_mem_reserved=max_mem_reserved.mean(),
    )


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
        "--num_inference_steps",
        type=int,
        default=None,
        required=False,
        help="the number of inference steps to run for the pipeline",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, required=False, help="the size of a batch"
    )

    args = parser.parse_args()

    print(
        f"{args.pipeline=} {args.model_id=} {args.runs=} {args.batch_size=} "
        f"{args.num_inference_steps=}"
    )

    start = time()
    pipeline = load_pipeline(args.pipeline, args.model_id)

    # Collect pipeline load metrics
    load_time = time() - start
    load_max_mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
    load_max_mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

    # Collect pipeline warmup metrics if stable-fast is enabled
    if os.getenv("SFAST", "").strip().lower() == "true":
        warmups = 3
        warmup_metrics = bench_pipeline(
            pipeline, args.batch_size, warmups, args.num_inference_steps
        )

    # Collect pipeline inference metrics
    metrics = bench_pipeline(
        pipeline, args.batch_size, args.runs, args.num_inference_steps
    )

    print("\n")
    print("----AGGREGATE METRICS----")
    print("\n")

    print(f"pipeline load time: {load_time:.3f}s")
    print(f"pipeline load max GPU memory allocated: {load_max_mem_allocated:.3f}GiB")
    print(f"pipeline load max GPU memory reserved: {load_max_mem_reserved:.3f}GiB")

    if os.getenv("SFAST", "").strip().lower() == "true":
        print(f"avg warmup inference time: {warmup_metrics.inference_time:.3f}s")
        print(
            f"avg warmup inference time per output: "
            f"{warmup_metrics.inference_time_per_output:.3f}s"
        )
        print(
            f"avg warmup inference max GPU memory allocated: "
            f"{warmup_metrics.max_mem_allocated:.3f}GiB"
        )
        print(
            f"avg warmup inference max GPU memory reserved: "
            f"{warmup_metrics.max_mem_reserved:.3f}GiB"
        )

    print(f"avg inference time: {metrics.inference_time:.3f}s")
    print(f"avg inference time per output: {metrics.inference_time_per_output:.3f}s")
    print(f"avg inference max GPU memory allocated: {metrics.max_mem_allocated:.3f}GiB")
    print(f"avg inference max GPU memory reserved: {metrics.max_mem_reserved:.3f}GiB")
