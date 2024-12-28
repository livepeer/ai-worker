import logging
import os
from typing import List, Union, Dict, Any, Optional
import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from dataclasses import dataclass
from enum import Enum
from huggingface_hub import file_download

logger = logging.getLogger(__name__)


class EmbeddingModelType(Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    TRANSFORMER = "transformer"
    INSTRUCTOR = "instructor"


@dataclass
class EmbeddingConfig:
    normalize: bool = True
    max_length: int = 512
    batch_size: int = 32

    def validate(self):
        """Validate embedding parameters"""
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


class EmbeddingPipeline(Pipeline):
    def __init__(self, model_id: str):
        """Initialize the Embedding Pipeline."""
        logger.info("Initializing embedding pipeline")

        self.model_id = model_id
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model")
        base_path = os.path.join(get_model_dir(), folder_name)

        # Find the actual model path
        self.local_model_path = self._find_model_path(base_path)

        if not self.local_model_path:
            raise ValueError(f"Could not find model files for {model_id}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get configuration from environment
        model_type = os.getenv("EMBEDDING_MODEL_TYPE", "sentence-transformer")
        self.model_type = EmbeddingModelType(model_type)
        self.max_length = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))

        logger.info(f"Loading embedding model: {model_id} of type {model_type}")

        try:
            if self.model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
                self.model = SentenceTransformer(self.local_model_path).to(self.device)
            elif self.model_type == EmbeddingModelType.INSTRUCTOR:
                self.model = INSTRUCTOR(self.local_model_path).to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    async def generate(
        self,
        texts: Union[str, List[str]],
        embedding_config: Optional[EmbeddingConfig] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate embeddings for input texts."""
        config = embedding_config or EmbeddingConfig()

        if isinstance(texts, str):
            texts = [texts]

        try:
            if self.model_type == EmbeddingModelType.INSTRUCTOR and instruction:
                texts = [f"{instruction} {text}" for text in texts]

            embeddings = self.model.encode(
                texts,
                batch_size=config.batch_size,
                normalize_embeddings=config.normalize,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            embeddings_list = embeddings.cpu().numpy().tolist()

            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": emb,
                        "index": i,
                    }
                    for i, emb in enumerate(embeddings_list)
                ],
                "model": self.model_id,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts),
                }
            }

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def __call__(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings with configuration."""
        try:
            config = EmbeddingConfig(**kwargs)
            config.validate()
            return await self.generate(texts, config)
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise

    def __str__(self):
        return f"EmbeddingPipeline(model_id={self.model_id})"
    
    def _find_model_path(self, base_path):
    # Check if the model files are directly in the base path
        if any(file.endswith('.bin') or file.endswith('.safetensors') for file in os.listdir(base_path)):
            return base_path

        # If not, look in subdirectories
        for root, dirs, files in os.walk(base_path):
            if any(file.endswith('.bin') or file.endswith('.safetensors') for file in files):
                return root
