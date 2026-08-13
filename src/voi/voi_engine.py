"""High-level orchestration module for the Value of Information (VoI) Engine.

Provides the primary API for calculating VoI scores and making communication
decision actions on single observations or batch datasets.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Union
import pandas as pd

from src.voi.decision_policy import DecisionAction, PolicyThresholds, evaluate_decision
from src.voi.normalization import VoIInputs
from src.voi.scoring import ScoringResult, VoIWeights, calculate_voi_score


@dataclass
class VoIResult:
    """Structured output result for a single observation evaluation."""

    novelty: float
    uncertainty: float
    task_relevance: float
    temporal_importance: float
    resource_cost: float
    raw_voi_score: float
    voi_score: float
    decision: DecisionAction
    timestamp: Optional[Any] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert result object to dictionary representation."""
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


class VoIEngine:
    """Value of Information Engine Orchestrator.

    Accepts normalized input variables, calculates VoI mathematical scores using
    configurable weights, and applies decision policy thresholds to determine
    communication actions.
    """

    def __init__(
        self,
        weights: Optional[VoIWeights] = None,
        thresholds: Optional[PolicyThresholds] = None,
        clip_inputs: bool = False,
    ):
        """Initialize the VoI Engine.

        Args:
            weights: Weight configuration. Defaults to equal baseline weights (0.20 each).
            thresholds: Policy thresholds configuration. Defaults to provisional thresholds.
            clip_inputs: If True, clip out-of-bounds input values to [0, 1] instead of
                         raising an error.
        """
        self.weights = weights if weights is not None else VoIWeights()
        self.thresholds = thresholds if thresholds is not None else PolicyThresholds()
        self.clip_inputs = clip_inputs

    def compute(
        self,
        novelty: float,
        uncertainty: float,
        task_relevance: float,
        temporal_importance: float,
        resource_cost: float,
        timestamp: Optional[Any] = None,
    ) -> VoIResult:
        """Compute VoI score and decision action for a single observation.

        Args:
            novelty: Normalized novelty N in [0, 1].
            uncertainty: Normalized prediction uncertainty U in [0, 1].
            task_relevance: Normalized task relevance R in [0, 1].
            temporal_importance: Normalized temporal importance T in [0, 1].
            resource_cost: Normalized resource/communication cost C in [0, 1].
            timestamp: Optional timestamp identifier for observation.

        Returns:
            VoIResult: Structured evaluation result object.
        """
        inputs = VoIInputs(
            novelty=novelty,
            uncertainty=uncertainty,
            task_relevance=task_relevance,
            temporal_importance=temporal_importance,
            resource_cost=resource_cost,
            clip=self.clip_inputs,
        )

        scoring_res: ScoringResult = calculate_voi_score(
            inputs=inputs, weights=self.weights, clip_output=True
        )

        decision: DecisionAction = evaluate_decision(
            voi_score=scoring_res.voi_score, thresholds=self.thresholds
        )

        metadata = {
            "weights": asdict(self.weights),
            "thresholds": asdict(self.thresholds),
        }

        return VoIResult(
            timestamp=timestamp,
            novelty=inputs.novelty,
            uncertainty=inputs.uncertainty,
            task_relevance=inputs.task_relevance,
            temporal_importance=inputs.temporal_importance,
            resource_cost=inputs.resource_cost,
            raw_voi_score=scoring_res.raw_voi_score,
            voi_score=scoring_res.voi_score,
            decision=decision,
            metadata=metadata,
        )

    def compute_batch(
        self, df_or_records: Union[pd.DataFrame, List[dict]]
    ) -> pd.DataFrame:
        """Compute VoI score and decision actions for a batch of observations.

        Args:
            df_or_records: Pandas DataFrame or list of dicts containing columns:
                           'novelty', 'uncertainty', 'task_relevance',
                           'temporal_importance', 'resource_cost', (and optional 'timestamp').

        Returns:
            pd.DataFrame: DataFrame containing input observations enriched with
                          'raw_voi_score', 'voi_score', and 'decision'.
        """
        if isinstance(df_or_records, list):
            df = pd.DataFrame(df_or_records)
        elif isinstance(df_or_records, pd.DataFrame):
            df = df_or_records.copy()
        else:
            raise TypeError(
                f"Expected DataFrame or list of dicts, got {type(df_or_records).__name__}"
            )

        results = []
        for _, row in df.iterrows():
            ts = row.get("timestamp", None)
            res = self.compute(
                novelty=row["novelty"],
                uncertainty=row["uncertainty"],
                task_relevance=row["task_relevance"],
                temporal_importance=row["temporal_importance"],
                resource_cost=row["resource_cost"],
                timestamp=ts,
            )
            results.append(res.to_dict())

        res_df = pd.DataFrame(results)

        # Retain any extra columns from input df (such as scenario label)
        for col in df.columns:
            if col not in res_df.columns:
                res_df[col] = df[col].values

        return res_df
