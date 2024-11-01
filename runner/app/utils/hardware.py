
from typing import Dict
from pydantic import BaseModel
import torch
import pynvml
import logging

logger = logging.getLogger(__name__)

class HardwareDetail(BaseModel):
    id: str
    name: str
    memory_total: int
    memory_free: int
    major: int
    minor: int

class HardwareStatDetail(BaseModel):
    id: str
    name: str
    memory_free: int
    memory_total: int
    utilization_compute: int
    utilization_memory: int

def get_cuda_devices() -> Dict[int, HardwareDetail]:
    devices = {}
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            devices[i] = HardwareDetail(
                id=uuid,
                name=name,
                memory_total=memory_info.total,
                memory_free=memory_info.free,
                major=major,
                minor=minor
            )
    except BaseException as e:
        logger.info(f"Error getting cuda devices: error={e}")
    finally:
        pynvml.nvmlShutdown()
    
    return devices

def get_cuda_stats() -> Dict[int, HardwareStatDetail]:
    stats = {}
    try:
        pynvml.nvmlInit()
        
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            stats[i] = HardwareStatDetail(
                id=uuid,
                name=name,
                memory_free=memory_info.free,
                memory_total=memory_info.total,
                utilization_compute=utilization.gpu,
                utilization_memory=utilization.memory
            )
    except BaseException as e:
        logger.info(f"Error getting cuda devices: error={e}")
    finally:
        pynvml.nvmlShutdown()

    return stats