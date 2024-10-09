from typing import Literal, Optional, List, Dict
from PIL import Image
from pydantic import BaseModel, Field

from StreamDiffusionWrapper import StreamDiffusionWrapper

from .interface import Transmorgrifier

class StreamKohakuParams(BaseModel):
    prompt: str = "anime drawing style"
    model_id: str = "KBlueLeaf/kohaku-v2.1"
    lora_dict: Optional[Dict[str, float]] = None
    use_lcm_lora: bool = True
    num_inference_steps: int = 50
    t_index_list: Optional[List[int]] = None
    t_index_ratio_list: Optional[List[float]] = [0.75, 0.9, 0.975]
    scale: float = 1.0
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"
    use_denoising_batch: bool = True
    enable_similar_image_filter: bool = True
    seed: int = 2
    guidance_scale: float = 1.2

    def __init__(self, **data):
        super().__init__(**data)
        if self.t_index_ratio_list is not None:
            self.t_index_list = [int(i * self.num_inference_steps) for i in self.t_index_ratio_list]

class StreamKohaku(Transmorgrifier):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = StreamKohakuParams(**params)
        self.pipe: Optional[StreamDiffusionWrapper] = None
        self.last_prompt = self.params.prompt

    def process_frame(self, image: Image.Image) -> Image.Image:
        first = self.pipe is None
        if self.pipe is None:
            self.pipe = StreamDiffusionWrapper(
                model_id_or_path=self.params.model_id,
                lora_dict=self.params.lora_dict,
                use_lcm_lora=self.params.use_lcm_lora,
                t_index_list=self.params.t_index_list,
                frame_buffer_size=1,
                width=image.width,
                height=image.height,
                warmup=10,
                acceleration=self.params.acceleration,
                do_add_noise=False,
                mode="img2img",
                # output_type="pt",
                enable_similar_image_filter=self.params.enable_similar_image_filter,
                similar_image_filter_threshold=0.98,
                use_denoising_batch=self.params.use_denoising_batch,
                seed=self.params.seed,
            )
            self.pipe.prepare(
                prompt=self.params.prompt,
                num_inference_steps=self.params.num_inference_steps,
                guidance_scale=self.params.guidance_scale,
            )

        if self.last_prompt != self.params.prompt:
            self.pipe.stream.update_prompt(self.params.prompt)
            self.last_prompt = self.params.prompt

        img_tensor = self.pipe.preprocess_image(image)
        img_tensor = self.pipe.stream.image_processor.denormalize(img_tensor)

        if first:
            for _ in range(self.pipe.batch_size):
                self.pipe(image=img_tensor)

        return self.pipe(image=img_tensor)

    def update_params(self, **params):
        new_params = StreamKohakuParams(**params)
        # reset the pipe if anything changed other than the prompt
        only_prompt = self.params.model_copy(update={'prompt': new_params.prompt})
        if new_params != only_prompt:
            self.pipe = None
            print(f"Reset diffuser for params change")
        self.params = new_params
