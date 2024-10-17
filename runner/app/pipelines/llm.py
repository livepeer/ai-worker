import asyncio
import logging
import os
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_max_memory
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput
from huggingface_hub import file_download

logger = logging.getLogger(__name__)

class LLMPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        folder_name = file_download.repo_folder_name(repo_id=model_id, repo_type="model")
        base_path = os.path.join(get_model_dir(), folder_name)
        
        # Find the actual model path
        self.local_model_path = self._find_model_path(base_path)
        
        if not self.local_model_path:
            raise ValueError(f"Could not find model files for {model_id}")

        use_8bit = os.getenv("USE_8BIT", "").strip().lower() == "true"
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "4096"))
        max_num_seqs = int(os.getenv("MAX_NUM_SEQS", "128"))
        mem_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.80"))

        # Get available GPU memory
        max_memory = get_max_memory()       
        logger.info(f"Available GPU memory: {max_memory.gpu_memory}")

        engine_args = AsyncEngineArgs(
            model=self.local_model_path,
            tokenizer=self.local_model_path,
            trust_remote_code=True,
            dtype="auto",  # This specifies BFloat16 precision, TODO: Check GPU capabilities to set best type
            kv_cache_dtype="auto",  # or "fp16" if you want to force it
            tensor_parallel_size=max_memory.num_gpus,
            max_num_batched_tokens=max_batch_size,
            gpu_memory_utilization=mem_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=False,
            enable_prefix_caching=True,
        )

        if use_8bit:
            engine_args.quantization = "bitsandbytes"
            logger.info("Using 8-bit quantization")
        else:
            logger.info("Using BFloat16 precision")

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        logger.info(f"Model loaded: {self.model_id}")
        logger.info(f"Using GPU memory utilization: {mem_utilization}")
        self.engine.start_background_loop()

    async def __call__(self, prompt: str, history: Optional[List[tuple]] = None, system_msg: Optional[str] = None, **kwargs) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        start_time = time.time()
    
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        if history:
            for user_msg, assistant_msg in history:
                conversation.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    conversation.append({"role": "assistant", "content": assistant_msg})
        conversation.append({"role": "user", "content": prompt})

        tokenizer = await self.engine.get_tokenizer()
        full_prompt = tokenizer.apply_chat_template(conversation, tokenize=False)

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 256),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
        )

        request_id = str(time.monotonic())
        results_generator = self.engine.generate(prompt=full_prompt, sampling_params=sampling_params, request_id=request_id)

        generated_tokens = 0
        first_token_time = None
        previous_text = ""

        try:
            async for request_output in results_generator:
                if first_token_time is None:
                    first_token_time = time.time()
                
                text = request_output.outputs[0].text
                new_text = text[len(previous_text):]
                generated_tokens += len(tokenizer.encode(new_text))
                
                yield new_text
                previous_text = text
                await asyncio.sleep(0)  # Allow other tasks to run
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

        end_time = time.time()

        # Calculate total tokens and timing
        prompt_tokens = len(tokenizer.encode(full_prompt))
        total_tokens = prompt_tokens + generated_tokens
        total_time = end_time - start_time
        generation_time = end_time - first_token_time if first_token_time else 0

        # Log benchmarking information
        logger.info(f"Generation completed:")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Prompt tokens: {prompt_tokens}")
        logger.info(f"  Generated tokens: {generated_tokens}")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info(f"  Time to first token: {(first_token_time - start_time):.2f} seconds")
        logger.info(f"  Generation time: {generation_time:.2f} seconds")
        logger.info(f"  Tokens per second: {total_tokens / generation_time:.2f}")

        yield {"tokens_used": total_tokens}

    def __str__(self):
        return f"LLMPipeline(model_id={self.model_id})"

    def _find_model_path(self, base_path):
        # Check if the model files are directly in the base path
        if any(file.endswith('.bin') or file.endswith('.safetensors') for file in os.listdir(base_path)):
            return base_path

        # If not, look in subdirectories
        for root, dirs, files in os.walk(base_path):
            if any(file.endswith('.bin') or file.endswith('.safetensors') for file in files):
                return root

        return None