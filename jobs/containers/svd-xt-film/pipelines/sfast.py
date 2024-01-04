from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig


def compile_model(model):
    config = CompilationConfig.Default()

    # xformers and Triton are suggested for achieving best performance.
    # It might be slow for Triton to generate, compile and fine-tune kernels.
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")
    # NOTE:
    # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
    # Disable Triton if you encounter this problem.
    try:
        import triton

        config.enable_triton = True
    except ImportError:
        print("Triton not installed, skip")

    model = compile(model, config)
    return model
