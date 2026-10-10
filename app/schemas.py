from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CycleValue(BaseModel):
    cycle: int
    model_config = ConfigDict(extra='allow')


class PredictRequest(BaseModel):
    engine_id: Optional[str] = None
    cycles: list[CycleValue] = Field(
        ...,
        min_length=1,
        json_schema_extra={
            'example': {
                'engine_id': 'engine-42',
                'cycles': [
                    {'cycle': 1, 'sensor_2': 641.8, 'sensor_3': 1589.7, 'sensor_4': 1410.0},
                ],
            }
        },
    )
    model_config = ConfigDict(extra='ignore')


class PredictResponse(BaseModel):
    predicted_rul: float
    alert: bool
    model_version: str
    model_config = ConfigDict(protected_namespaces=())
