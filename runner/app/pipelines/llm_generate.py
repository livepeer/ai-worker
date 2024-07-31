import logging
import os
from threading import Thread
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from huggingface_hub import file_download, hf_hub_download

logger = logging.getLogger(__name__)


class LLMGeneratePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.history = [] 

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

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

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

        conversation = []
        
        for user,  assistant in self.history:
            consersation.extend([{"role":"user", "content": user}, {"role":"assistant", "contanstant": assistant}])
        conversation.append({"role":"user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        # Update generation config
        generate_kwargs = dict(
            input_ids= input_ids,
            streamer= streamer,
            max_new_tokens= max_tokens,
            do_sample= True,
            temperature=temperature,
            eos_token_id=self.terminators,
        )
        
        if temperature == 0:
            generate_kwargs['do_sample'] = Fals


        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text) 

        # Calculate tokens used
        tokens_used = len(outputs)
        complete_response = "".join(outputs)

        return {
            "response": complete_response,
            "tokens_used": tokens_used
        }

    def __str__(self) -> str:
        return f"LLMPipeline model_id={self.model_id}"
