from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
import logging

logger = logging.getLogger(__name__)


def compile_model(model):
    config = CompilationConfig.Default()

    # xformers and Triton are suggested for achieving best performance.
    # It might be slow for Triton to generate, compile and fine-tune kernels.
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        logger.info("xformers not installed, skip")
    # NOTE:
    # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
    # Disable Triton if you encounter this problem.
    try:
        import triton

        config.enable_triton = True
    except ImportError:
        logger.info("Triton not installed, skip")

    model = compile(model, config)
    return model
