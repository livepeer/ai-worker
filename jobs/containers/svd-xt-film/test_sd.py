# huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --cache-dir cache2
# huggingface-cli download stabilityai/stable-diffusion-xl-refiner-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --cache-dir cache2

from diffusers import DiffusionPipeline
import torch
import time
import numpy as np
import gc

begin = time.time()

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="./cache2",
)
base.to("cuda")

# base.enable_model_cpu_offload()

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="./cache2",
)
refiner.to("cuda")
# refiner.enable_model_cpu_offload()

end = time.time()

load_time = end - begin

prompt = ["A majestic lion jumping from a big stone at night"] * 4

# prompt = [
#     "A majestic lion jumping from a big stone at night",
#     "A panda cooking in the kitchen",
# ]

runs = 2

run_times = []


# for _ in range(runs):
#     begin = time.time()

#     image = base(
#         prompt=prompt,
#         num_inference_steps=40,
#         denoising_end=0.8,
#         output_type="latent",
#     ).images

#     images = refiner(
#         prompt=prompt,
#         num_inference_steps=40,
#         denoising_start=0.8,
#         image=image,
#     ).images

#     # for idx, image in enumerate(images):
#     #     image.save(f"output/sdxl-{idx}-out.png")

#     end = time.time()

#     run_times.append(end - begin)

#     # flush_memory()

# for idx, run_time in enumerate(run_times):
#     print(f"run time {idx}: {run_time:.3f}s")

# avg_run_time = np.average(run_times)
# print(f"throughput: {avg_run_time / len(prompt)} s/img")

print(f"model load time: {load_time:.3f}s")

peak_mem_allocated = torch.cuda.max_memory_allocated()
peak_mem_reserved = torch.cuda.max_memory_reserved()
print(f"peak GPU memory allocated: {peak_mem_allocated / 1024**3:.3f}GiB")
print(f"peak GPU memory reserved: {peak_mem_reserved / 1024**3:.3f}GiB")
