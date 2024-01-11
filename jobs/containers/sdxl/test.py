from diffusers import DiffusionPipeline
import torch
import time

begin = time.time()

base_pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
base = DiffusionPipeline.from_pretrained(
    base_pipe_id, torch_dtype=torch.float16, variant="fp16", cache_dir="./cache"
)
base.to("cuda")

end = time.time()

base_model_load_time = end - begin

begin = time.time()

base.load_lora_weights(
    "./cache/ral-friedegg-sdxl-lora.safetensors", adapter_name="ral-friedegg"
)

end = time.time()

lora_load_time = end - begin

prompt = "a jigsaw puzzle made of ral-friedegg with one piece missing"

begin = time.time()

lora_scale = 0.9
generator = torch.manual_seed(0)
kwargs = {"num_inference_steps": 40, "cross_attention_kwargs": {"scale": lora_scale}}
image = base(prompt, **kwargs).images[0]

end = time.time()

inference_time = end - begin

image.save("output/output_0.png")

base.load_lora_weights(
    "./cache/piecesglass-sdxl-lora.safetensors", adapter_name="pieces-glass"
)

prompt = "a jigsaw puzzle made of pieces of glass"

image = base(prompt, **kwargs).images[0]

image.save("output/output_1.png")

print(f"base model load time: {base_model_load_time:.3f}s")
print(f"lora load time: {lora_load_time:.3f}s")
print(f"inference time: {inference_time:.3f}is")

peak_mem_allocated = torch.cuda.max_memory_allocated()
peak_mem_reserved = torch.cuda.max_memory_reserved()
print(f"peak GPU memory allocated: {peak_mem_allocated / 1024**3:.3f}GiB")
print(f"peak GPU memory reserved: {peak_mem_reserved / 1024**3:.3f}GiB")
