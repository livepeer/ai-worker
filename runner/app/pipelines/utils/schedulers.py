import importlib, json, logging
from typing import Tuple, Union, List, Dict, get_type_hints, get_origin, get_args

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
    elif pipeline == "ImageToImagePipeline":
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
        type_hints = get_type_hints(sch_cls.__init__)
        for arg in set_sch["args"]:
            if not is_type_accepted(set_sch["args"][arg], type_hints[arg]):
                return None, None, f"params not in correct format: {arg}={set_sch['args'][arg]}. Type should be {type_hints[arg]}, provided {type(set_sch['args'][arg])}" 
              
        return sch_cls, set_sch["args"], ""

    except BaseException as e:
        return None, None, f"scheduler not available: {e}"
    
def is_type_accepted(value, type_hint) -> bool:
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Case 1: If no origin, it’s a basic type
    if origin is None:
        return isinstance(value, type_hint)

    # Case 2: If it's a Union (e.g., Union[int, str])
    if origin is Union:
        return any(is_type_accepted(value, arg) for arg in args)

    # Case 3: If it's an Optional type (e.g., Optional[int] which is Union[int, None])
    if origin is Union and type(None) in args:
        return value is None or any(is_type_accepted(value, arg) for arg in args if arg is not type(None))

    # Case 4: If it’s a generic collection like List[int]
    if origin in {list, List}:
        return isinstance(value, list) and all(is_type_accepted(v, args[0]) for v in value)

    if origin in {dict, Dict}:
        key_type, value_type = args
        return isinstance(value, dict) and all(
            is_type_accepted(k, key_type) and is_type_accepted(v, value_type) for k, v in value.items()
        )  