from dataclasses import dataclass, field
from typing import Dict

import pandas as pd
from qate.core.model import Market

from pipeline.framework import StageContext


@dataclass
class Context(StageContext):
    market: Market
    target: str
    source_df: pd.DataFrame = None  # aligned data (read-only)
    features: Dict[str, pd.Series] = field(default_factory=dict)  # accumulated features
