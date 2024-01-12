from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import torch
import time

begin = time.time()

repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
cache_dir = "./cache"
pipeline = StableVideoDiffusionPipeline.from_pretrained(
    repo_id, cache_dir=cache_dir, variant="fp16", torch_dtype=torch.float16
)
pipeline.to("cuda")

end = time.time()

load_time = end - begin

# Load the conditioning image
# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
# )
# image = [image.resize((1024, 576))]
# image *= 2

# generator = torch.manual_seed(42)


# begin = time.time()
# frames = pipeline(
#     image,
#     decode_chunk_size=8,
#     generator=generator,
#     output_type="np",
# ).frames
# end = time.time()

# print(type(frames))
# print(frames.shape)

# run_time = end - begin
# print(f"run time: {run_time:.3f}s")

print(f"model load time: {load_time:.3f}s")

peak_mem_allocated = torch.cuda.max_memory_allocated()
peak_mem_reserved = torch.cuda.max_memory_reserved()
print(f"peak GPU memory allocated: {peak_mem_allocated / 1024**3:.3f}GiB")
print(f"peak GPU memory reserved: {peak_mem_reserved / 1024**3:.3f}GiB")
