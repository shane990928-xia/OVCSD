"""Track skill usage and outcomes by task type."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Skill dropout (create with/without contrast for attribution)
# --------------------------------------------------------------------------- #
def apply_skill_dropout(retrieved: dict, rate: float, rng) -> Tuple[dict, List[str]]:
    """Randomly drop skills from a retrieval result to create attribution contrast.

    ``retrieved`` is the dict returned by SkillsOnlyMemory.retrieve (keys
    'general_skills' / 'task_specific_skills', each a list of skill dicts with
    'skill_id'). Returns (new_retrieved, kept_skill_ids). Never drops ALL skills
    of a non-empty category (keep at least one so the prompt stays meaningful).
    """
    if rate <= 0.0:
        kept = _all_skill_ids(retrieved)
        return retrieved, kept

    new = dict(retrieved)
    for key in ("general_skills", "task_specific_skills"):
        skills = list(retrieved.get(key, []) or [])
        if not skills:
            continue
        keep_mask = [rng.random() >= rate for _ in skills]
        if not any(keep_mask):
            keep_mask[int(rng.integers(len(skills)))] = True  # keep >=1
        new[key] = [s for s, k in zip(skills, keep_mask) if k]
    return new, _all_skill_ids(new)


def _all_skill_ids(retrieved: dict) -> List[str]:
    ids = []
    for key in ("general_skills", "task_specific_skills"):
        for s in retrieved.get(key, []) or []:
            sid = s.get("skill_id")
            if sid:
                ids.append(str(sid))
    return ids


# --------------------------------------------------------------------------- #
# Utility attribution (U(s) and de-confounded δ(s))
# --------------------------------------------------------------------------- #
@dataclass
class SkillUtilityTracker:
    """Accumulates A^cf per skill via a with/without contrast (leave-one-out).

    δ(s) = mean over task types of [ mean(A^cf | s injected, type t)
                                     − mean(A^cf | s NOT injected, type t) ].
    Skill dropout provides samples without injection for comparison within each task type.

    Feed one call per decision point with A^cf, task_type, and the injected skills.
    """

    _with: Dict[tuple, List[float]] = field(default_factory=lambda: defaultdict(list))     # (sid, ttype) -> a_cf when injected
    _without: Dict[tuple, List[float]] = field(default_factory=lambda: defaultdict(list))  # (sid, ttype) -> a_cf when not injected
    _type_pool: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))  # ttype -> all a_cf (category-spread metric)
    _universe: set = field(default_factory=set)

    def register_universe(self, skill_ids: Sequence[str]) -> None:
        """Pre-register the candidate skill set so WITHOUT samples accrue for
        skills not injected at a decision point."""
        self._universe |= {str(s) for s in skill_ids}

    def observe_decision(self, a_cf: float, task_type: str, skill_ids: Sequence[str]) -> None:
        a_cf = float(a_cf)
        ttype = str(task_type)
        injected = {str(s) for s in skill_ids}
        self._universe |= injected
        self._type_pool[ttype].append(a_cf)
        for sid in self._universe:
            if sid in injected:
                self._with[(sid, ttype)].append(a_cf)
            else:
                self._without[(sid, ttype)].append(a_cf)

    def _types_for(self, sid: str) -> set:
        return {t for (s, t) in self._with if s == sid}

    def u_score(self, skill_id: str) -> Optional[float]:
        sid = str(skill_id)
        vals = [a for (s, _), lst in self._with.items() if s == sid for a in lst]
        return (sum(vals) / len(vals)) if vals else None

    def delta_score(self, skill_id: str) -> Optional[float]:
        sid = str(skill_id)
        per_type = []
        for ttype in self._types_for(sid):
            w = self._with.get((sid, ttype))
            wo = self._without.get((sid, ttype))
            if not w or not wo:
                continue
            per_type.append(sum(w) / len(w) - sum(wo) / len(wo))
        return (sum(per_type) / len(per_type)) if per_type else None

    def sample_count(self, skill_id: str) -> int:
        sid = str(skill_id)
        return sum(len(lst) for (s, _), lst in self._with.items() if s == sid)

    def contrast_count(self, skill_id: str) -> int:
        sid = str(skill_id)
        return sum(len(lst) for (s, _), lst in self._without.items() if s == sid)

    def all_skill_ids(self) -> List[str]:
        return sorted({s for (s, _) in self._with})

    def summary(self) -> dict:
        deltas = [d for d in (self.delta_score(s) for s in self.all_skill_ids()) if d is not None]
        us = [u for u in (self.u_score(s) for s in self.all_skill_ids()) if u is not None]
        u_spread = (max(us) - min(us)) if us else 0.0
        delta_spread = (max(deltas) - min(deltas)) if deltas else 0.0
        cat_means = [(sum(p) / len(p)) for p in self._type_pool.values() if p]
        cat_spread = (max(cat_means) - min(cat_means)) if cat_means else 0.0
        return {
            "cpsd/skill_util_spread": float(u_spread),
            "cpsd/skill_util_category_spread": float(cat_spread),
            "cpsd/skill_delta_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
            "cpsd/skill_delta_spread": float(delta_spread),
            "cpsd/skill_tracked_count": float(len(self.all_skill_ids())),
        }


# --------------------------------------------------------------------------- #
# Prune & weight decisions (by δ(s), not U(s))
# --------------------------------------------------------------------------- #
def prune_and_weight(
    tracker: SkillUtilityTracker,
    *,
    prune_threshold: float = 0.0,
    min_samples: int = 10,
    weight_by_delta: bool = True,
) -> Tuple[set, Dict[str, float]]:
    """Return (pruned_skill_ids, retrieval_weights).

    A skill is pruned if it has enough samples AND δ(s) ≤ threshold (useless /
    harmful after de-confounding). Retrieval weight = sigmoid-like positive
    monotone in δ(s) (skills with higher causal utility rank first). Skills with
    too few samples are left unpruned with weight 1.0 (undecided).
    """
    pruned = set()
    weights: Dict[str, float] = {}
    for sid in tracker.all_skill_ids():
        n = tracker.sample_count(sid)
        delta = tracker.delta_score(sid)
        if delta is None or n < min_samples:
            weights[sid] = 1.0
            continue
        if delta <= prune_threshold:
            pruned.add(sid)
            weights[sid] = 0.0
        else:
            weights[sid] = (1.0 + delta) if weight_by_delta else 1.0
    return pruned, weights


def prune_metrics(pruned: set, weights: Dict[str, float], contrast_samples: int) -> dict:
    kept_weights = [w for sid, w in weights.items() if sid not in pruned]
    return {
        "cpsd/skill_pruned_count": float(len(pruned)),
        "cpsd/skill_dropout_contrast_samples": float(contrast_samples),
        "cpsd/skill_kept_weight_mean": float(sum(kept_weights) / len(kept_weights)) if kept_weights else 0.0,
    }
