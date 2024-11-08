from .interface import Pipeline

def load_pipeline(name: str, **params) -> Pipeline:
    if name == "streamdiffusion":
        from .streamdiffusion import StreamDiffusion
        return StreamDiffusion(**params)
    elif name == "liveportrait":
        from .liveportrait import LivePortrait
        return LivePortrait(**params)
    elif name == "segment_anything_2":
        from .segment_anything_2 import Sam2Live
        return Sam2Live(**params)
    raise ValueError(f"Unknown pipeline: {name}")
