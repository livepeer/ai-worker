import logging
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class TextSentimentAnalysisPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {}

        torch_device = get_torch_device()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            cache_dir=get_model_dir(),
            **kwargs,
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=get_model_dir())

        self.ldm = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )

    def __call__(self, text: str, **kwargs):
        return self.ldm(text, **kwargs)

    def __str__(self) -> str:
        return f"TextSentimentAnalysisPipeline model_id={self.model_id}"
