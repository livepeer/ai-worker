import logging
import os

import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device, DetectionFrame
from huggingface_hub import file_download
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from typing import List
from PIL import Image, ImageDraw, ImageFont
from app.utils.errors import InferenceError

logger = logging.getLogger(__name__)

def annotate_image(input_image, detections, labels, font_size, font):
    draw = ImageDraw.Draw(input_image)
    bounding_box_color = (255, 255, 0)  # Bright Yellow for bounding box
    text_color = (0, 0, 0)               # Black for text
    for box, label in zip(detections["boxes"], labels):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=bounding_box_color, width=3)
        # Place label above the bounding box
        draw.text((x1, y1 - font_size - 5), label, fill=text_color, font=font)  # Adjust y position
    return input_image



class ObjectDetectionPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {}

        self.torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load fp16 variant if fp16 safetensors files are found in cache
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if self.torch_device != "cpu" and has_fp16_variant:
            logger.info("ObjectDetectionPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("ObjectDetectionPipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        self.object_detection_model = AutoModelForObjectDetection.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=get_model_dir(),
            **kwargs,
        ).to(self.torch_device)        

        #try to use fast image processor if possible
        # new fast image processors are being added slowly to transformers, fallback to the default one if not available
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_id, 
                cache_dir=get_model_dir(),
                use_fast=True
            )
        except:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_id, 
                cache_dir=get_model_dir()
            )

        # Load a font (default font is used here; you can specify your own path for a TTF file)
        self.font_size = 24
        self.font = ImageFont.load_default(size=self.font_size)


    def __call__(self, frames: List[DetectionFrame], confidence_threshold: float = 0.6, return_annotated_video: bool = False, **kwargs) -> str:
        try:
            annotated_frames = []
            confidence_scores_all_frames = []
            labels_all_frames = []
            detection_boxes_all_frames = []
            pts_all_frames = []

            for frame in frames:
                # Process frame and add annotations
                inputs = self.image_processor(images=frame.image, return_tensors="pt").to(self.torch_device)
                with torch.no_grad():
                    outputs = self.object_detection_model(**inputs)

                target_sizes = torch.tensor([frame.image.size[::-1]])
                results = self.image_processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=confidence_threshold,
                    target_sizes=target_sizes
                )[0]

                final_labels = []
                confidence_scores = []

                detections = {"boxes": results["boxes"].cpu().numpy()}

                for label_id, score in zip(results["labels"].cpu().numpy(),results["scores"].cpu().numpy()):
                    final_labels.append(self.object_detection_model.config.id2label[label_id])
                    confidence_scores.append(round(score, 3))

                if return_annotated_video:
                    annotated_frame = annotate_image(
                        input_image=frame,
                        detections=detections,
                        labels=final_labels,
                        font_size=self.font_size,
                        font=self.font
                    )
                    annotated_frames.append(annotated_frame)
                
                # List of detections, confidence scores, labels, and pts for each frame
                confidence_scores_all_frames.append(confidence_scores)
                labels_all_frames.append(final_labels)
                detection_boxes_all_frames.append(detections["boxes"].tolist())
                pts_all_frames.append(float(frame.pts * frame.time_base))

            return annotated_frames, confidence_scores_all_frames, labels_all_frames, detection_boxes_all_frames, pts_all_frames

        except Exception as e:
            raise InferenceError(original_exception=e)

    def __str__(self) -> str:
        return f"ObjectDetectionPipeline model_id={self.model_id}"