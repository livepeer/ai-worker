"""Contains utility functions for hardware information."""

from typing import Dict
from pydantic import BaseModel
import logging
from app.utils.nvml_manager import nvml_manager

logger = logging.getLogger(__name__)


class GpuBaseInfo(BaseModel):
    """Model for general GPU information."""

    id: str
    name: str
    memory_total: int
    memory_free: int


class GpuComputeInfo(GpuBaseInfo):
    """Model for detailed GPU compute information."""

    major: int
    minor: int


class GpuUtilizationInfo(GpuBaseInfo):
    """Model for real-time GPU utilization statistics."""

    utilization_compute: int
    utilization_memory: int


class GpuInfo(GpuComputeInfo, GpuUtilizationInfo):
    """Model for full CUDA device information."""

    pass


def retrieve_cuda_info() -> Dict[int, GpuInfo]:
    """Retrieve CUDA device information.

    Returns:
        CUDA device information.
    """
    devices = {}
    if not nvml_manager.initialized:
        logger.warning("NVML is not initialized.")
        return devices

    try:
        for i in range(nvml_manager.pynvml.nvmlDeviceGetCount()):
            handle = nvml_manager.pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = nvml_manager.pynvml.nvmlDeviceGetUUID(handle)
            name = nvml_manager.pynvml.nvmlDeviceGetName(handle)
            memory_info = nvml_manager.pynvml.nvmlDeviceGetMemoryInfo(handle)
            major, minor = nvml_manager.pynvml.nvmlDeviceGetCudaComputeCapability(
                handle
            )
            utilization_rates = nvml_manager.pynvml.nvmlDeviceGetUtilizationRates(
                handle
            )
            devices[i] = GpuInfo(
                id=uuid,
                name=name,
                memory_total=memory_info.total,
                memory_free=memory_info.free,
                major=major,
                minor=minor,
                utilization_compute=utilization_rates.gpu,
                utilization_memory=utilization_rates.memory,
            )
    except nvml_manager.pynvml.NVMLError as e:
        logger.warning(f"Failed to retrieve CUDA device information: {e}")
    return devices


def get_gpu_info() -> Dict[int, GpuComputeInfo]:
    """Get detailed GPU compute information.

    Returns:
        The detailed GPU compute information.
    """
    basic_info = retrieve_cuda_info()
    return {
        i: GpuComputeInfo(
            id=info.id,
            name=info.name,
            memory_total=info.memory_total,
            memory_free=info.memory_free,
            major=info.major,
            minor=info.minor,
        )
        for i, info in basic_info.items()
    }


def get_gpu_stats() -> Dict[int, GpuUtilizationInfo]:
    """Get real-time GPU utilization statistics.

    Returns:
        The real-time GPU utilization statistics.
    """
    basic_info = retrieve_cuda_info()
    return {
        i: GpuUtilizationInfo(
            id=info.id,
            name=info.name,
            memory_total=info.memory_total,
            memory_free=info.memory_free,
            utilization_compute=info.utilization_compute,
            utilization_memory=info.utilization_memory,
        )
        for i, info in basic_info.items()
    }
