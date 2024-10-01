from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthCheck(BaseModel):
    status: str = "OK"


@router.get("/health", operation_id="health", response_model=HealthCheck)
@router.get("/health/", response_model=HealthCheck, include_in_schema=False)
def health() -> HealthCheck:
    return HealthCheck(status="OK")
