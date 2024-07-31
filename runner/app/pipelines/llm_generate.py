import logging
import os
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from huggingface_hub import file_download, hf_hub_download

logger = logging.getLogger(__name__)


class LLMGeneratePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {
            "cache_dir": get_model_dir()
        }
        self.device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)

        # Check for fp16 variant
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if self.device != "cpu" and has_fp16_variant:
            logger.info("LLMGeneratePipeline loading fp16 variant for %s", model_id)
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, **kwargs).to(self.device)

        # Set up generation config
        self.generation_config = self.model.generation_config

        # Optional: Add optimizations
        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        if sfast_enabled:
            logger.info(
                "LLMGeneratePipeline will be dynamically compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model
            self.model = compile_model(self.model)

    def __call__(self, prompt: str, system_msg: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        if system_msg:
            input_text = f"{system_msg}\n\n{prompt}"
        else:
            input_text = prompt

        input_ids = self.tokenizer.encode(
            input_text, return_tensors="pt").to(self.device)

        # Update generation config
        gen_kwargs = {}
        if temperature is not None:
            gen_kwargs['temperature'] = temperature
        if max_tokens is not None:
            gen_kwargs['max_new_tokens'] = max_tokens

        # Merge generation config with provided kwargs
        gen_kwargs = {**self.generation_config.to_dict(), **gen_kwargs, **kwargs}

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                **gen_kwargs
            )

        # Decode the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Calculate tokens used
        tokens_used = len(output[0])

        return {
            "response": response.strip(),
            "tokens_used": tokens_used
        }

    def __str__(self) -> str:
        return f"LLMPipeline model_id={self.model_id}"
