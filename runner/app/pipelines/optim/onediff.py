"""This module provides a function to enable optimization for the pipeline using OneFlow and OneDiffx.

For more information, see the OneDiffx project on GitHub: https://github.com/siliconflow/onediffx
"""

import logging
from onediffx import compile_pipe
import oneflow as flow

logger = logging.getLogger(__name__)

def enable_onediff(pipe):
    """Optimize the pipeline with OneFlow and OneDiffx.

    Args:
        pipe: The pipeline to be optimized.

    Returns:
        The optimized pipeline.
    """
    try:
        # Compile the UNet component of the pipeline with OneFlow
       

        # Automatically map devices for the entire pipeline using OneDiffx
        pipe = compile_pipe (pipe)
        logger.info("Pipeline optimized with OneFlow and OneDiffx.")
    
    except Exception as e:
        logger.error(f"Failed to optimize pipeline with OneFlow and OneDiffx: {str(e)}")
    
    return pipe
