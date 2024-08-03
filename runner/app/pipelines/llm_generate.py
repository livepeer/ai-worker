import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from huggingface_hub import file_download, snapshot_download
from threading import Thread

logger = logging.getLogger(__name__)

class LLMGeneratePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {
            "cache_dir": get_model_dir(),
            "local_files_only": True
        }
        self.device = get_torch_device()
   
        # Generate the correct folder name
        folder_path = file_download.repo_folder_name(repo_id=model_id, repo_type="model")
        self.local_model_path = os.path.join(get_model_dir(), folder_path)
        self.checkpoint_dir = snapshot_download(model_id, cache_dir=get_model_dir(), local_files_only=True)

        logger.info(f"Local model path: {self.local_model_path}")
        logger.info(f"Directory contents: {os.listdir(self.local_model_path)}")

        # Check for fp16 variant
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(self.local_model_path)
            for fname in files
        )
        if self.device != "cpu" and has_fp16_variant:
            logger.info("LLMGeneratePipeline loading fp16 variant for %s", model_id)
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
        elif self.device != "cpu":
            kwargs["torch_dtype"] = torch.bfloat16

        logger.info(f"Loading model {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        
        # Load the model configuration
        config = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).config
        
        # Initialize empty weights
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)

        # Prepare for distributed setup
        num_gpus = torch.cuda.device_count()
        max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory // 1024**3}GiB" for i in range(num_gpus)}
        max_memory["cpu"] = "24GiB"  # Adjust based on your system's RAM
        
        logger.info(f"Max memory configuration: {max_memory}")

        # Load and dispatch the model
        self.model = load_checkpoint_and_dispatch(
            self.model,
            self.checkpoint_dir,
            device_map="auto",
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],  # Adjust based on your model architecture
            dtype=kwargs.get("torch_dtype", torch.float32),
            offload_folder="offload",  # Optional: specify a folder for offloading
            offload_state_dict=True,  # Optional: offload state dict to CPU
        )

        logger.info(f"Model loaded and distributed. Device map: {self.model.hf_device_map}")

        # Set up generation config
        self.generation_config = self.model.generation_config

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Optional: Add optimizations
        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        if sfast_enabled:
            logger.info(
                "LLMGeneratePipeline will be dynamically compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model
            self.model = compile_model(self.model)

    async def __call__(self, prompt: str, history: Optional[List[tuple]] = None, system_msg: Optional[str] = None, **kwargs) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        if history:
            for user, assistant in history:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        conversation.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(input_ids)

        max_new_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = self.generation_config.to_dict()
        generate_kwargs.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        })

        thread = Thread(target=self.model_generate_wrapper, kwargs=generate_kwargs)
        thread.start()

        total_tokens = 0
        try:
            for text in streamer:
                total_tokens += 1
                yield text
                await asyncio.sleep(0)  # Allow other tasks to run
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise

        input_length = input_ids.size(1)
        yield {"tokens_used": input_length + total_tokens}

    def model_generate_wrapper(self, **kwargs):
        try:
            logger.debug("Entering model.generate")
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                self.model.generate(**kwargs)
            logger.debug("Exiting model.generate")
        except Exception as e:
            logger.error(f"Error in model.generate: {str(e)}", exc_info=True)
            raise

    def __str__(self):
        return f"LLMGeneratePipeline(model_id={self.model_id})"