import numpy
import pandas as pd
from typing import List, Union
from pydantic import BaseModel

class FeatureIn(BaseModel):
    data_type: str = None
    season: int = None
    team: Union[List[str], str] = "all"
