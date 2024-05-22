import numpy
import pandas as pd
from typing import List, Union
from pydantic import BaseModel, ConfigDict

class FeatureIn(BaseModel):
    data_type: str = None
    season: int = None

class FeatureOut(BaseModel):
    column_names: list
    best_params: object
    best_score: float
