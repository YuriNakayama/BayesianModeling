# test functions
import numpy as np
import pandas as pd
from typing import Union
from dataclasses import dataclass, field

@dataclass
class IndicatorConfig:
    unit_price: int = 10_000
    target_ratio: float = 0.12

class Indicator:
    def __init__(self, config: IndicatorConfig, scores: pd.Series, flgs: pd.Series) -> None:
        self.config = config
        self.scores = scores
        self.flgs = flgs
        
    def precision_at_k(self, k: float) -> float:
        user_count = len(self.scores)
        target_user_count = int(user_count * k)
        
        sorted_index = self.scores.sort_values().index
        target_user_flgs = self.flgs.loc[sorted_index].head(target_user_count)
        positive_count = target_user_flgs.sum()
        
        return positive_count / len(target_user_flgs) if len(target_user_flgs) > 0 else 0
        
        
    
    def deterrent_effect(self) -> float:
        return self.precision_at_k(self.config.target_ratio) * self.config.unit_price
