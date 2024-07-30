import logging
import os
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device

logger = logging.getLogger(__name__)


class LLMGeneratePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = get_torch_device()

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=get_model_dir())
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=get_model_dir(),
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Set up generation config
        self.generation_config = GenerationConfig.from_pretrained(model_id)
        self.generation_config.max_length = 2048  # Adjust as needed

    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                generation_config=self.generation_config,
                **kwargs
            )

        # Decode the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response

    def __str__(self) -> str:
        return f"LLMGeneratePipeline model_id={self.model_id}"
