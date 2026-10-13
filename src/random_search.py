"""Baseline 1: uniform random search (Phase 6).

Draws candidates from the Phase 5 generator. Phase 8 selects the reference-anchored
`local_perturbation` generator a priori; the `uniform_random` generator remains available for
reference-only diagnostics. The local generator uses training/reference feature rows only and adds
bounded noise; it never sees labels, the target model, predictions or failure outcomes. with the explicit run seed, and
evaluates them through the shared `FeedbackManager` (-> FailureValidator -> QueryBudgetedModel).
It uses no labels, no test data, no target-model internals and no feedback: the score it receives is
logged but never used to choose the next candidate. Nothing is handicapped: it gets the same budget,
validator, duplicate policy and proposal cap as every other method.

Because Phase 5 candidate i of a seed is independent of batch size, the evaluated sequence does not
depend on `batch_size`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .candidate_generator import GENERATOR_NAMES, make_generator
from .search_base import SearchEngine
from .search_space import SearchSpaceConfig


@dataclass(frozen=True)
class RandomSearchConfig:
    generator: str = "uniform_random"
    batch_size: int = 64

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "RandomSearchConfig":
        r = cfg["search"]["random"]
        c = cls(r["generator"], int(r["batch_size"]))
        c.validate()
        return c

    def validate(self) -> None:
        bad = []
        if self.generator not in GENERATOR_NAMES: bad.append(f"generator must be one of {GENERATOR_NAMES}")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            bad.append("batch_size must be an integer >= 1")
        if bad:
            raise ValueError("Invalid search.random config: " + "; ".join(bad))


class RandomSearch(SearchEngine):
    method = "random"

    def __init__(self, space, validator, query_budget, seed, common, method_cfg: Optional[RandomSearchConfig] = None, *,
                 space_config: Optional[SearchSpaceConfig] = None, paper_eligible_context: bool = False):
        method_cfg = method_cfg or RandomSearchConfig()
        method_cfg.validate()
        super().__init__(space, validator, query_budget, seed, common, method_cfg,
                         paper_eligible_context=paper_eligible_context)
        if method_cfg.generator == "local_perturbation" and space_config is None:
            raise ValueError("local_perturbation needs `space_config` (Phase 5 SearchSpaceConfig)")
        # anchors = reference FEATURE rows only (labels are never read)
        self.generator = make_generator(method_cfg.generator, space, space_config or SearchSpaceConfig(),
                                        anchors=validator.ref.X if method_cfg.generator == "local_perturbation" else None)

    def _search(self, session) -> str:
        index = 0
        while not session.stopped:
            n = min(self.cfg.batch_size, session.proposals_left)
            cands = self.generator.generate(n, seed=self.seed, start_index=index)
            index += n
            session.note_generated(len(cands))
            session.evaluate(cands)
        return session.stop_reason

    def extra_reproducibility(self) -> Dict[str, Any]:
        return {"generator": self.generator.describe()}
