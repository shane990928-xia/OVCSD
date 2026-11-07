"""Pure utilities for student-first teacher intervention.

These helpers summarize step-level rollout batches into trajectory/group
records so the trainer can decide whether to run extra with-skill teacher
rollouts. They never modify rewards, advantages, or actor-loss tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class TrajectorySummary:
    group_uid: str
    traj_uid: str
    reward: float
    length: float
    success: bool
    row_indices: tuple[int, ...]
    task_score: float = 0.0


@dataclass(frozen=True)
class StudentGroupStats:
    group_uid: str
    reward_mean: float
    reward_best: float
    reward_min: float
    reward_std: float
    success_frac: float
    all_failed: bool
    traj_count: int
    traj_uids: tuple[str, ...]
    task_score_mean: float = 0.0
    task_score_best: float = 0.0
    task_score_std: float = 0.0


@dataclass(frozen=True)
class TeacherGroupStats:
    group_uid: str
    reward_best: float
    reward_mean: float
    success: bool
    best_traj_uid: Optional[str]
    traj_count: int
    successful_traj_uids: tuple[str, ...] = ()
    traj_uids: tuple[str, ...] = ()
    task_score_best: float = 0.0
    task_score_mean: float = 0.0


@dataclass(frozen=True)
class TeacherInterventionDecision:
    group_uid: str
    teacher_used: bool
    teacher_success: bool
    utility: float
    utility_mode: str
    student_reward_mean: float
    student_reward_best: float
    teacher_reward_best: float
    best_teacher_traj_uid: Optional[str]
    successful_teacher_traj_uids: tuple[str, ...] = ()
    teacher_traj_uids: tuple[str, ...] = ()
    teacher_task_score_best: float = 0.0
    student_task_score_mean: float = 0.0


def _as_array(values, *, dtype=None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _field(non_tensor_batch: Mapping[str, object], key: str, default=None):
    return non_tensor_batch[key] if key in non_tensor_batch else default


def _string_at(values: np.ndarray, idx: int, default: str) -> str:
    if values.size == 0:
        return default
    value = values[idx]
    if value is None:
        return default
    return str(value)


def _float_at(values: np.ndarray, idx: int, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    try:
        return float(values[idx])
    except Exception:
        return default


def summarize_trajectories(
    batch,
    *,
    success_reward_threshold: float = 10.0,
    group_key: str = "uid",
    traj_key: str = "traj_uid",
    reward_key: str = "episode_rewards",
    length_key: str = "episode_lengths",
    task_score_key: str = "episode_task_scores",
) -> List[TrajectorySummary]:
    """Collapse step rows into one record per trajectory.

    ``episode_rewards`` and ``episode_lengths`` are repeated on every active
    step row. Deduplicating by ``traj_uid`` avoids length-weighting long
    trajectories.
    """
    ntb = batch.non_tensor_batch
    rewards = _as_array(_field(ntb, reward_key), dtype=object)
    n = len(rewards)
    if n == 0:
        return []

    group_values = _as_array(_field(ntb, group_key), dtype=object)
    traj_values = _as_array(_field(ntb, traj_key), dtype=object)
    length_values = _as_array(_field(ntb, length_key), dtype=object)
    task_score_values = _as_array(_field(ntb, task_score_key), dtype=object)

    by_traj: Dict[str, dict] = {}
    order: List[str] = []
    for i in range(n):
        traj_uid = _string_at(traj_values, i, default=f"traj-{i}")
        if traj_uid not in by_traj:
            by_traj[traj_uid] = {
                "group_uid": _string_at(group_values, i, default=f"group-{i}"),
                "reward": _float_at(rewards, i),
                "length": _float_at(length_values, i),
                "task_score": _float_at(task_score_values, i),
                "row_indices": [],
            }
            order.append(traj_uid)
        else:
            by_traj[traj_uid]["reward"] = _float_at(rewards, i, by_traj[traj_uid]["reward"])
            by_traj[traj_uid]["length"] = _float_at(length_values, i, by_traj[traj_uid]["length"])
            by_traj[traj_uid]["task_score"] = _float_at(task_score_values, i, by_traj[traj_uid]["task_score"])
        by_traj[traj_uid]["row_indices"].append(i)

    out: List[TrajectorySummary] = []
    for traj_uid in order:
        item = by_traj[traj_uid]
        reward = float(item["reward"])
        out.append(
            TrajectorySummary(
                group_uid=str(item["group_uid"]),
                traj_uid=traj_uid,
                reward=reward,
                length=float(item["length"]),
                success=reward >= success_reward_threshold,
                row_indices=tuple(int(i) for i in item["row_indices"]),
                task_score=float(item["task_score"]),
            )
        )
    return out


def compute_student_group_stats(
    batch,
    *,
    success_reward_threshold: float = 10.0,
) -> Dict[str, StudentGroupStats]:
    trajectories = summarize_trajectories(batch, success_reward_threshold=success_reward_threshold)
    grouped: Dict[str, List[TrajectorySummary]] = {}
    for traj in trajectories:
        grouped.setdefault(traj.group_uid, []).append(traj)

    stats: Dict[str, StudentGroupStats] = {}
    for group_uid, items in grouped.items():
        rewards = np.asarray([t.reward for t in items], dtype=float)
        successes = np.asarray([t.success for t in items], dtype=float)
        task_scores = np.asarray([t.task_score for t in items], dtype=float)
        stats[group_uid] = StudentGroupStats(
            group_uid=group_uid,
            reward_mean=float(rewards.mean()),
            reward_best=float(rewards.max()),
            reward_min=float(rewards.min()),
            reward_std=float(rewards.std()),
            success_frac=float(successes.mean()),
            all_failed=bool(successes.sum() == 0),
            traj_count=int(len(items)),
            traj_uids=tuple(t.traj_uid for t in items),
            task_score_mean=float(task_scores.mean()),
            task_score_best=float(task_scores.max()),
            task_score_std=float(task_scores.std()),
        )
    return stats


def compute_teacher_acquisition_scores(
    student_stats: Mapping[str, StudentGroupStats],
    *,
    signal: str = "task_score",
    reward_scale: float = 10.0,
    floor: float = 0.05,
    hardness_power: float = 1.0,
    uncertainty_weight: float = 0.25,
) -> Dict[str, float]:
    """Return positive, threshold-free scores for budgeted teacher queries."""
    scale = max(abs(float(reward_scale)), 1e-8)
    floor = max(float(floor), 0.0)
    power = max(float(hardness_power), 0.0)
    uncertainty_weight = max(float(uncertainty_weight), 0.0)

    scores: Dict[str, float] = {}
    for group_uid, stats in student_stats.items():
        if signal == "task_score":
            mean = float(np.clip(stats.task_score_mean, 0.0, 1.0))
            std = float(np.clip(stats.task_score_std, 0.0, 1.0))
        elif signal == "reward":
            mean = float(np.clip(stats.reward_mean / scale, 0.0, 1.0))
            std = float(np.clip(stats.reward_std / scale, 0.0, 1.0))
        else:
            raise ValueError(f"Unknown teacher acquisition signal: {signal}")
        difficulty = max(1.0 - mean, 0.0) ** power
        scores[str(group_uid)] = float(floor + difficulty + uncertainty_weight * std)
    return scores


def select_teacher_groups(
    student_stats: Mapping[str, StudentGroupStats],
    *,
    trigger: str = "hard_only",
    success_reward_threshold: float = 10.0,
    probability: float = 0.0,
    budget: Optional[int] = None,
    acquisition_signal: str = "task_score",
    acquisition_floor: float = 0.05,
    acquisition_hardness_power: float = 1.0,
    acquisition_uncertainty_weight: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Choose student groups eligible for a with-skill teacher rollout."""
    if trigger == "always":
        return list(student_stats.keys())
    if trigger == "hard_only":
        return [
            group_uid
            for group_uid, stats in student_stats.items()
            if stats.all_failed or stats.reward_best < success_reward_threshold
        ]
    if trigger == "mean_below":
        return [
            group_uid
            for group_uid, stats in student_stats.items()
            if stats.reward_mean < success_reward_threshold
        ]
    if trigger == "probability":
        if probability <= 0:
            return []
        if probability >= 1:
            return list(student_stats.keys())
        rng = rng if rng is not None else np.random.default_rng()
        return [group_uid for group_uid in student_stats if bool(rng.random() < probability)]
    if trigger == "budgeted_soft":
        group_uids = list(student_stats.keys())
        if not group_uids:
            return []
        query_budget = len(group_uids) if budget is None else max(int(budget), 0)
        query_budget = min(query_budget, len(group_uids))
        if query_budget <= 0:
            return []
        if query_budget == len(group_uids):
            return group_uids

        scores = compute_teacher_acquisition_scores(
            student_stats,
            signal=acquisition_signal,
            reward_scale=success_reward_threshold,
            floor=acquisition_floor,
            hardness_power=acquisition_hardness_power,
            uncertainty_weight=acquisition_uncertainty_weight,
        )
        weights = np.asarray([scores[str(group_uid)] for group_uid in group_uids], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(group_uids), dtype=np.float64)
        probabilities = weights / weights.sum()
        rng = rng if rng is not None else np.random.default_rng()
        selected = set(
            int(index)
            for index in rng.choice(len(group_uids), size=query_budget, replace=False, p=probabilities)
        )
        return [group_uid for index, group_uid in enumerate(group_uids) if index in selected]
    raise ValueError(f"Unknown teacher intervention trigger: {trigger}")


def compute_teacher_group_stats(
    batch,
    *,
    success_reward_threshold: float = 10.0,
) -> Dict[str, TeacherGroupStats]:
    trajectories = summarize_trajectories(batch, success_reward_threshold=success_reward_threshold)
    grouped: Dict[str, List[TrajectorySummary]] = {}
    for traj in trajectories:
        grouped.setdefault(traj.group_uid, []).append(traj)

    stats: Dict[str, TeacherGroupStats] = {}
    for group_uid, items in grouped.items():
        rewards = np.asarray([t.reward for t in items], dtype=float)
        task_scores = np.asarray([t.task_score for t in items], dtype=float)
        best_idx = int(rewards.argmax())
        best = items[best_idx]
        stats[group_uid] = TeacherGroupStats(
            group_uid=group_uid,
            reward_best=float(best.reward),
            reward_mean=float(rewards.mean()),
            success=bool(any(t.success for t in items)),
            best_traj_uid=best.traj_uid,
            traj_count=int(len(items)),
            successful_traj_uids=tuple(t.traj_uid for t in items if t.success),
            traj_uids=tuple(t.traj_uid for t in items),
            task_score_best=float(task_scores.max()),
            task_score_mean=float(task_scores.mean()),
        )
    return stats


def compute_teacher_intervention_decisions(
    student_stats: Mapping[str, StudentGroupStats],
    teacher_stats: Mapping[str, TeacherGroupStats],
    *,
    selected_group_uids: Optional[Iterable[str]] = None,
    utility_mode: str = "best_minus_student_mean",
    task_score_utility: bool = False,
) -> Dict[str, TeacherInterventionDecision]:
    """Combine student/teacher summaries into per-group gates and utility.

    ``task_score_utility`` uses dense terminal ``task_score`` for gating
    utility only. It does not change the on-policy student reward or
    advantage computation.
    """
    selected = set(selected_group_uids) if selected_group_uids is not None else set(teacher_stats.keys())
    decisions: Dict[str, TeacherInterventionDecision] = {}

    for group_uid in selected:
        if group_uid not in student_stats:
            continue
        student = student_stats[group_uid]
        teacher = teacher_stats.get(group_uid)
        teacher_used = teacher is not None
        teacher_success = bool(teacher.success) if teacher is not None else False
        teacher_best = float(teacher.reward_best) if teacher is not None else 0.0
        teacher_task_best = float(teacher.task_score_best) if teacher is not None else 0.0

        if task_score_utility:
            teacher_signal = teacher_task_best
            student_mean_signal = student.task_score_mean
            student_best_signal = student.task_score_best
        else:
            teacher_signal = teacher_best
            student_mean_signal = student.reward_mean
            student_best_signal = student.reward_best

        if utility_mode == "best_minus_student_mean":
            utility = teacher_signal - student_mean_signal
        elif utility_mode == "best_minus_student_best":
            utility = teacher_signal - student_best_signal
        else:
            raise ValueError(f"Unknown utility_mode: {utility_mode}")

        decisions[group_uid] = TeacherInterventionDecision(
            group_uid=group_uid,
            teacher_used=teacher_used,
            teacher_success=teacher_success,
            utility=float(utility) if teacher_used else 0.0,
            utility_mode=utility_mode,
            student_reward_mean=student.reward_mean,
            student_reward_best=student.reward_best,
            teacher_reward_best=teacher_best,
            best_teacher_traj_uid=teacher.best_traj_uid if teacher is not None else None,
            successful_teacher_traj_uids=teacher.successful_traj_uids if teacher is not None else (),
            teacher_traj_uids=teacher.traj_uids if teacher is not None else (),
            teacher_task_score_best=teacher_task_best,
            student_task_score_mean=student.task_score_mean,
        )

    return decisions


def summarize_teacher_intervention(
    decisions: Mapping[str, TeacherInterventionDecision],
) -> Dict[str, float]:
    if not decisions:
        return {
            "skill_int/teacher_groups": 0.0,
            "skill_int/teacher_used_frac": 0.0,
            "skill_int/teacher_success_frac": 0.0,
            "skill_int/U_mean": 0.0,
            "skill_int/U_positive_frac": 0.0,
            "skill_int/teacher_successful_trajs": 0.0,
            "skill_int/teacher_successful_trajs_per_group": 0.0,
        }

    vals = list(decisions.values())
    used = np.asarray([d.teacher_used for d in vals], dtype=float)
    success = np.asarray([d.teacher_success for d in vals], dtype=float)
    utility = np.asarray([d.utility for d in vals], dtype=float)
    successful_trajs = np.asarray([len(d.successful_teacher_traj_uids) for d in vals], dtype=float)
    return {
        "skill_int/teacher_groups": float(len(vals)),
        "skill_int/teacher_used_frac": float(used.mean()),
        "skill_int/teacher_success_frac": float(success.mean()),
        "skill_int/U_mean": float(utility.mean()),
        "skill_int/U_positive_frac": float((utility > 0).mean()),
        "skill_int/teacher_successful_trajs": float(successful_trajs.sum()),
        "skill_int/teacher_successful_trajs_per_group": float(successful_trajs.mean()),
    }
