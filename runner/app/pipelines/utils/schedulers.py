import importlib, json, logging
from typing import Tuple, get_type_hints

logger = logging.getLogger(__name__)

text_to_image_presets = {
    "DPM++ 2M": { "name": "DPMSolverMultistepScheduler", "args": {} },
    "DPM++ 2M Karras": { "name": "DPMSolverMultistepScheduler", "args": { "use_karras_sigmas": True } },
    "DPM++ 2M SDE": { "name": "DPMSolverMultistepScheduler", "args": { "algorithm_type": "sde-dpmsolver++" } },
    "DPM++ 2M SDE Karras": { "name": "DPMSolverMultistepScheduler", "args": { "use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++" } },
    "DPM++ 2S a": { "name": "DPMSolverSinglestepScheduler", "args": {} },
    "DPM++ 2S a Karras": { "name": "DPMSolverSinglestepScheduler", "args": { "use_karras_sigmas": True } },
    "DPM++ SDE": { "name": "DPMSolverSinglestepScheduler", "args": {} },
    "DPM++ SDE Karras": { "name": "DPMSolverSinglestepScheduler", "args": { "use_karras_sigmas": True } },
    "DPM2": { "name": "KDPM2DiscreteScheduler", "args": {} },
    "DPM2 Karras": { "name": "KDPM2DiscreteScheduler", "args": { "use_karras_sigmas": True } },
    "DPM2 a": { "name": "KDPM2AncestralDiscreteScheduler", "args": {} },
    "DPM2 a Karras": { "name": "KDPM2AncestralDiscreteScheduler", "args": { "use_karras_sigmas": True } },
    "Euler": { "name": "EulerDiscreteScheduler", "args": {} },
    "Euler a": { "name": "EulerAncestralDiscreteScheduler", "args": {} },
    "Euler flow match": { "name": "FlowMatchEulerDiscreteScheduler", "args": {} },
    "Huen": { "name": "HeunDiscreteScheduler", "args": {} },
    "LMS": { "name": "LMSDiscreteScheduler", "args": {} },
    "LMS Karras": { "name": "LMSDiscreteScheduler", "args": { "use_karras_sigmas": True } }
}


def load_scheduler_presets(pipeline: str) -> any:
    if pipeline == "TextToImagePipeline":
        return text_to_image_presets
    else:
        return {}

def create_scheduler(scheduler: str, presets: dict) -> Tuple[object, dict, str]:
    """
    creates scheduler from provided settings (name/args).  Presets available for convenience setting main schedulers
    """
    set_sch = json.loads(scheduler)
    set_sch_name = set_sch.get("name", None)
    if set_sch_name in presets:
        set_sch["name"] = presets[set_sch_name]["name"]
        set_sch["args"] = presets[set_sch_name]["args"] | set_sch["args"]
    
    try:
        sch_cls = getattr(importlib.import_module("diffusers"), set_sch["name"])
        #convert params to correct type.  The from_config method passes the **kwargs provided directly
        #to the __init__ method of the scheduler (diffusers/src/diffusers/configuration_utils.py)
        type_hints = get_type_hints(sch_cls.__init__)
        for arg in set_sch["args"]:
            try:
                set_sch["args"][arg] = type_hints[arg](set_sch["args"][arg])
            except:
                return None, None, f"params not in correct format: {arg}={set_sch['args'][arg]}"
            
        return sch_cls, set_sch["args"], ""

    except BaseException as e:
        return None, None, f"scheduler not available: {e}"
    
    