# src/api/pydantic_models.py
from typing import Any
from pydantic import BaseModel
from typing import Dict

class RiskRequest(BaseModel):
    """
    The input must include all features the model expects.
    Example:
    {
        "features": {
            "Recency_woe": 1.23,
            "TxnCount_woe": 3.1,
            ...
        }
    }
    """
    features: Dict[str, Any]

class RiskResponse(BaseModel):
    """
    The output: a simple JSON with the predicted probability.
    """
    probability: float