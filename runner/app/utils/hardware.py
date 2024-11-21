"""Contains utility functions for hardware information."""

from typing import Dict, Any
from pydantic import BaseModel
import logging
import pynvml
from app.utils.nvml_manager import nvml_manager

logger = logging.getLogger(__name__)


class BaseHardwareDetail(BaseModel):
    """Base model for GPU information."""

    id: str
    name: str
    memory_total: int
    memory_free: int


class HardwareDetail(BaseHardwareDetail):
    """Model for GPU information."""

    major: int
    minor: int


class HardwareStatDetail(BaseHardwareDetail):
    """Model for real-time GPU statistics."""

    utilization_compute: int
    utilization_memory: int


def retrieve_cuda_info(
    cuda_version: bool = False, utilization: bool = False
) -> Dict[int, Dict[str, Any]]:
    """Retrieve CUDA information.

    Args:
        cuda_version: Whether to retrieve CUDA version information.
        utilization: Whether to retrieve GPU utilization information.

    Returns:
        CUDA information.
    """
    nvml_manager.initialize()
    devices = {}
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        device_info = {
            "id": uuid,
            "name": name,
            "memory_total": memory_info.total,
            "memory_free": memory_info.free,
        }

        if cuda_version:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            device_info["major"] = major
            device_info["minor"] = minor

        if utilization:
            utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            device_info["utilization_compute"] = utilization_rates.gpu
            device_info["utilization_memory"] = utilization_rates.memory

        devices[i] = device_info

    return devices


def get_gpu_info() -> Dict[int, HardwareDetail]:
    """Get GPU information.

    Returns:
        The GPU information.
    """
    basic_info = retrieve_cuda_info(cuda_version=True)
    return {
        i: HardwareDetail(
            id=info["id"],
            name=info["name"],
            memory_total=info["memory_total"],
            memory_free=info["memory_free"],
            major=info["major"],
            minor=info["minor"],
        )
        for i, info in basic_info.items()
    }


def get_gpu_stats() -> Dict[int, HardwareStatDetail]:
    """Get real-time GPU statistics.

    Returns:
        The real-time GPU statistics.
    """
    basic_info = retrieve_cuda_info(utilization=True)
    return {
        i: HardwareStatDetail(
            id=info["id"],
            name=info["name"],
            memory_total=info["memory_total"],
            memory_free=info["memory_free"],
            utilization_compute=info["utilization_compute"],
            utilization_memory=info["utilization_memory"],
        )
        for i, info in basic_info.items()
    }
