from pipelines.svd_film import StableVideoDiffusionFILMPipeline
from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def main():
    pipeline = StableVideoDiffusionFILMPipeline(cache_dir="./cache")

    image = "input/1.png"
    pipeline(output_path="output/output.mp4", image=image)


if __name__ == "__main__":
    main()
