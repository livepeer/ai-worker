from pipelines.svd_film import StableVideoDiffusionFILMPipeline
from PIL import PngImagePlugin
import time
import torch

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def main():
    svd_config = {"sfast": True, "quantize": False, "no_fusion": False}
    pipeline = StableVideoDiffusionFILMPipeline(
        cache_dir="./cache", svd_config=svd_config
    )

    image = "input/1.png"

    runs = 3

    run_times = []
    for _ in range(runs):
        begin = time.time()
        pipeline(output_path="output/output.mp4", image=image)
        end = time.time()

        run_times.append(end - begin)

    for idx, run_time in enumerate(run_times):
        print(f"run time {idx}: {run_time:.3f}s")

    peak_mem = torch.cuda.max_memory_allocated()
    print(f"peak GPU memory: {peak_mem / 1024**3:.3f}GiB")


if __name__ == "__main__":
    main()
