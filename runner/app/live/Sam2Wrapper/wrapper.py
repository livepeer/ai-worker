# Copied from StreamDiffusion/utils/wrapper.py
import logging
from typing import List, Optional, Tuple

import torch
from sam2.build_sam import build_sam2_camera_predictor
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from PIL import Image
from typing import Optional, Dict

MODEL_MAPPING = {
    "facebook/sam2-hiera-tiny": {
        "config": "sam2_hiera_t.yaml",
        "checkpoint": "sam2_hiera_tiny.pt"
    },
    "facebook/sam2-hiera-small": {
        "config": "sam2_hiera_s.yaml",
        "checkpoint": "sam2_hiera_small.pt"
    },
    "facebook/sam2-hiera-base": {
        "config": "sam2_hiera_b.yaml",
        "checkpoint": "sam2_hiera_base.pt"
    },
    "facebook/sam2-hiera-large": {
        "config": "sam2_hiera_l.yaml",
        "checkpoint": "sam2_hiera_large.pt"
    }
}

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Initialize Hydra to load the configuration
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
    
config_path = "/workspaces/ai-worker/runner/models/sam2_configs"
sam2_checkpoint = "/models/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

class Sam2Wrapper:
    def __init__(
        self,
        model_id_or_path: str,
        device: Optional[str] = None,
        **kwargs
    ):
        self.model_id = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Code ripped out of sam2.build_sam.build_sam2_camera_predictor to appease Hydra
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name=model_cfg)

            hydra_overrides = [
                "++model._target_=sam2.sam2_camera_predictor.SAM2CameraPredictor",
            ]
            hydra_overrides_extra = [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                "++model.fill_hole_area=8",
            ]
            hydra_overrides.extend(hydra_overrides_extra)

            cfg = compose(config_name=model_cfg, overrides=hydra_overrides)
            OmegaConf.resolve(cfg)

            #Load the model
            model = instantiate(cfg.model, _recursive_=True)
            load_checkpoint(model, sam2_checkpoint, self.device)
            model.to(self.device)
            model.eval()

            # Set the model in memory
            self.predictor = model

    def __call__(
        self, 
        image: Image.Image,
        **kwargs
    ) -> Tuple[List[Image.Image], List[Optional[bool]]]:
        pass

    def __str__(self) -> str:
        return f"Sam2Wrapper model_id={self.model_id}"
        

def load_checkpoint(model, ckpt_path, device):
    if ckpt_path is not None:

        sd = torch.load(ckpt_path, map_location=device)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(f"Missing keys: {missing_keys}")
            raise RuntimeError("Missing keys while loading checkpoint.")
        if unexpected_keys:
            logging.error(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError("Unexpected keys while loading checkpoint.")
        logging.info("Loaded checkpoint successfully.")

def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")