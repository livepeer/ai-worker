"""This module provides a function to enable OneDiff optimization for the pipeline.

For more information, see the OneDiff project on GitHub: https://github.com/siliconflow/onediff
"""

import logging
from typing import Any
from onediffx import compile_pipe

logger = logging.getLogger(__name__)

def compile_model(pipe):
    """Compile the pipeline with OneDiff optimization.

    Args:
        pipe (Pipeline): The pipeline to be optimized.

    Returns:
        Pipeline: The optimized pipeline.
    """
    config = CompilationConfig.Default()
    # OneDiff optimization with the OneFlow backend for best performance.
    logger.info("OneFlow backend is now active...")
    
    
    try:
        pipe = compile_pipe(pipe)
    except Exception as e:
        logger.error(f"Failed to compile pipeline with OneDiff: {e}")
        raise

    return pipe
