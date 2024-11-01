from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
import os

from app.utils.hardware import get_cuda_devices, get_cuda_stats, HardwareDetail, HardwareStatDetail

router = APIRouter()
class HardwareInformation(BaseModel):
    pipeline: str
    model_id: str
    gpu_info: Dict[int, HardwareDetail]

class HardwareStats(BaseModel):
    pipeline: str
    model_id: str
    gpu_stats: Dict[int, HardwareStatDetail]

@router.get("/hardware/info", operation_id="hardware_info", response_model=HardwareInformation)
async def hardware_info() -> HardwareInformation:
    
    return HardwareInformation(
        pipeline=os.environ["PIPELINE"],
        model_id=os.environ["MODEL_ID"],
        gpu_info=get_cuda_devices()
        )

@router.get("/hardware/stats", operation_id="hardware_stats", response_model=HardwareStats)
async def hardware_stats() -> HardwareStats:
    return HardwareStats(
        pipeline=os.environ["PIPELINE"],
        model_id=os.environ["MODEL_ID"],
        gpu_stats=get_cuda_stats()
    )