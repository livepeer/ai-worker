"""This module provides a function to enable onediff optimization for the pipeline.

For more information, see the DeepCache project on GitHub: https://github.com/siliconflow/onediff
"""

import logging

from onediffx import compile_pipe

logger = logging.getLogger(__name__)


def compile_model(pipe):
    """Compile the pipeline with onediff optimization.

    Args:
        pipe: The pipeline to be optimized.

    Returns:
        The optimized pipeline.
    """
    # oneflow compiler are suggested for achieving best performance.
    print("Oneflow backend is now active...")
    pipe = compile_pipe(pipe)
    return pipe
