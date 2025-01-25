from dataclasses import dataclass
from typing import List

@dataclass
class AttributionScore:
    index: int
    score: float

@dataclass
class LimeResult:
    attributions_l: List[AttributionScore]
    attributions_r: List[AttributionScore]
    match_score: float
    lime_intercept: float
    lime_pred_score: float
    lime_match_score: float

@dataclass
class LimeResultPair:
    attributions: List[AttributionScore]
    match_score: float
    lime_intercept: float
    lime_pred_score: float
    lime_match_score: float

@dataclass
class PairSegment:
    index_l: int
    index_r: int
    score: float
    match_score_diff: float
    del_left: bool
