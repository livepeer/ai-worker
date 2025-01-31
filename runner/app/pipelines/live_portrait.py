# coding: utf-8
from liveportrait.core.config.argument_config import ArgumentConfig
from liveportrait.core.config.inference_config import InferenceConfig
from liveportrait.core.config.crop_config import CropConfig
from liveportrait.core.live_portrait_pipeline import LivePortraitPipeline
from app.pipelines.base import Pipeline
import logging

logger = logging.getLogger(__name__)

class Inference(Pipeline):
    def __init__(self):
        self.args = ArgumentConfig()

    def __call__(self, source_image=None, driving_info=None) -> any:
        """Run the live portrait inference pipeline"""
        self.args.source_image = source_image
        self.args.driving_info = driving_info
        
        # Partial initialization of target_class fields with kwargs
        def _partial_fields(target_class, kwargs):
            return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

        # Specify configs for inference
        inference_cfg = _partial_fields(InferenceConfig, self.args.__dict__)
        crop_cfg = _partial_fields(CropConfig, self.args.__dict__)

        # Initialize the live portrait pipeline
        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )

        # Run the pipeline
        wfp = live_portrait_pipeline.execute(self.args)
        return wfp
