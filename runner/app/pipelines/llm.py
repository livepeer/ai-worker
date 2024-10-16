import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

from vllm import LLM, SamplingParams
from vllm.utils import InferenceRequest
from vllm.model_executor.parallel_utils import get_gpu_memory

logger = logging.getLogger(__name__)


class LLMPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.local_model_path = os.path.join(get_model_dir(), model_id)

        use_8bit = os.getenv("USE_8BIT", "").strip().lower() == "true"
        max_batch_size = os.getenv("MAX_BATCH_SIZE", "4096")
        max_num_seqs = os.getenv("MAX_NUM_SEQS", "256")
        mem_utilization = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")

        if use_8bit:
            quantization = "int8"
            logger.info("Using 8-bit quantization")
        else:
            quantization = "float16"  # Default to FP16
            logger.info("Using default FP16 precision")

        # Get available GPU memory
        gpu_memory = get_gpu_memory()
        logger.info(f"Available GPU memory: {gpu_memory}")

        # Initialize vLLM with more specific parameters
        self.llm = LLM(
            model=self.local_model_path,
            quantization=quantization,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=len(gpu_memory),  # Use all available GPUs
            max_num_batched_tokens=max_batch_size,  # Adjust based on your needs
            max_num_seqs=max_num_seqs,  # Adjust based on your needs
            gpu_memory_utilization=mem_utilization,  # Adjust GPU memory utilization
        )

        logger.info(f"Model loaded: {self.model_id}")
        logger.info(f"Using tensor parallelism across {len(gpu_memory)} GPUs")

    async def __call__(self, prompt: str, history: Optional[List[tuple]] = None, system_msg: Optional[str] = None, **kwargs) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        if history:
            conversation.extend(history)
        conversation.append({"role": "user", "content": prompt})

        # Apply chat template
        full_prompt = self.llm.get_tokenizer().apply_chat_template(conversation, tokenize=False)

        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
        )

        request_id = 0
        request = InferenceRequest(request_id, full_prompt, sampling_params)

        total_tokens = 0
        async for output in self.llm.generate_stream(request):
            if output.outputs:
                generated_text = output.outputs[0].text
                total_tokens += len(generated_text)
                yield generated_text
                await asyncio.sleep(0)  # Allow other tasks to run

        input_length = len(self.llm.get_tokenizer().encode(full_prompt))
        yield {"tokens_used": input_length + total_tokens}

    def __str__(self):
        return f"LLMPipeline(model_id={self.model_id})"
