"""Counterfactual first-divergence span selection for student-first CPSD."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class StudentFirstCFSpanPair:
    pair_id: str
    group_uid: str
    teacher_traj_uid: str
    student_traj_uid: str
    teacher_row_idx: int
    student_row_idx: int
    divergence_turn: int
    teacher_step: int
    student_step: int
    utility: float
    teacher_task_id: Optional[str] = None
    student_task_id: Optional[str] = None
    teacher_action: Optional[str] = None
    student_action: Optional[str] = None
    teacher_observation: Optional[str] = None
    student_observation: Optional[str] = None
    teacher_reward: Optional[float] = None
    student_reward: Optional[float] = None
    teacher_task_score: Optional[float] = None
    student_task_score: Optional[float] = None
    group_utility: Optional[float] = None
    teacher_response_hash: Optional[str] = None
    student_response_hash: Optional[str] = None
    # teacher turn-rows AFTER the divergence (the successful suffix to
    # distill via L_suffix), plus divergence diagnostics.
    suffix_teacher_rows: tuple = ()
    critical_divergence_ratio: float = 0.0
    successful_suffix_length: int = 0


def _all_diverging_turns(teacher_turns: Sequence[Optional[str]], student_turns: Sequence[Optional[str]]) -> List[int]:
    return [
        k
        for k in range(min(len(teacher_turns), len(student_turns)))
        if teacher_turns[k] is not None
        and student_turns[k] is not None
        and teacher_turns[k] != student_turns[k]
    ]


def _first_diverging_turn(teacher_turns: Sequence[Optional[str]], student_turns: Sequence[Optional[str]]) -> int:
    for k in range(min(len(teacher_turns), len(student_turns))):
        if teacher_turns[k] is not None and student_turns[k] is not None and teacher_turns[k] != student_turns[k]:
            return k
    return -1


def _num_rows(batch) -> int:
    if hasattr(batch, "batch") and "responses" in batch.batch:
        return int(batch.batch["responses"].shape[0])
    for values in getattr(batch, "non_tensor_batch", {}).values():
        try:
            return len(values)
        except Exception:
            continue
    return 0


def _value_at(values, idx: int, default=None):
    if values is None:
        return default
    try:
        return values[idx]
    except Exception:
        return default


def _str_at(values, idx: int, default: str) -> str:
    value = _value_at(values, idx, default)
    return default if value is None else str(value)


def _int_at(values, idx: int, default: int) -> int:
    try:
        return int(_value_at(values, idx, default))
    except Exception:
        return default


def _normalized_text(value) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def cf_action_at(batch, idx: int) -> Optional[str]:
    """Return the projected environment action, excluding hidden reasoning."""
    ntb = getattr(batch, "non_tensor_batch", {}) or {}
    raw_response = _value_at(ntb.get("model_response"), idx, None)
    if raw_response is not None:
        text = str(raw_response).lower()
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag, start_idx + len(start_tag)) if start_idx >= 0 else -1
        if start_idx >= 0 and end_idx >= 0:
            text = text[start_idx + len(start_tag) : end_idx]
        else:
            for tag in ("search", "answer"):
                start_tag = f"<{tag}>"
                end_tag = f"</{tag}>"
                start_idx = text.find(start_tag)
                end_idx = text.find(end_tag, start_idx + len(start_tag)) if start_idx >= 0 else -1
                if start_idx >= 0 and end_idx >= 0:
                    content = _normalized_text(text[start_idx + len(start_tag) : end_idx])
                    return f"<{tag}>{content}</{tag}>" if content else None
            think_start = text.find("<think>")
            think_end = text.find("</think>", think_start + len("<think>")) if think_start >= 0 else -1
            if think_start >= 0 and think_end >= 0:
                text = text[:think_start] + text[think_end + len("</think>") :]
        return _normalized_text(text)

    return None


def cf_observation_at(batch, idx: int) -> Optional[str]:
    ntb = getattr(batch, "non_tensor_batch", {}) or {}
    for key in ("observation", "anchor_obs"):
        value = _normalized_text(_value_at(ntb.get(key), idx, None))
        if value is not None:
            return value
    return None


def cf_response_hash_at(batch, idx: int) -> Optional[str]:
    responses = getattr(batch, "batch", {}).get("responses")
    if responses is None or idx >= int(responses.shape[0]):
        return None
    response = responses[idx].detach().cpu().contiguous()
    digest = hashlib.sha1()
    digest.update(str(tuple(response.shape)).encode("ascii"))
    digest.update(str(response.dtype).encode("ascii"))
    digest.update(response.numpy().tobytes())
    return digest.hexdigest()[:16]


def _episode_reward(batch, rows: Sequence[int]) -> Optional[float]:
    ntb = getattr(batch, "non_tensor_batch", {}) or {}
    values = ntb.get("episode_rewards")
    for row in rows:
        value = _value_at(values, row, None)
        if value is None:
            continue
        try:
            reward = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(reward):
            return reward
    return None


def _episode_task_score(batch, rows: Sequence[int]) -> Optional[float]:
    ntb = getattr(batch, "non_tensor_batch", {}) or {}
    values = ntb.get("episode_task_scores")
    for row in rows:
        value = _value_at(values, row, None)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            return score
    return None


def _states_align(student_batch, student_idx: int, teacher_batch, teacher_idx: int) -> bool:
    student_observation = cf_observation_at(student_batch, student_idx)
    teacher_observation = cf_observation_at(teacher_batch, teacher_idx)
    if student_observation is None or teacher_observation is None:
        return True
    return student_observation == teacher_observation


def _task_identity(batch, idx: int) -> Optional[str]:
    ntb = getattr(batch, "non_tensor_batch", {}) or {}
    instruction = _value_at(ntb.get("instruction_text"), idx, None)
    if instruction is not None:
        text = str(instruction).strip()
        if text:
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            return f"instruction:{digest}"
    for key in ("task_id", "session", "extra.gamefile", "seed"):
        value = _value_at(ntb.get(key), idx, None)
        if value is not None:
            return f"{key}:{value}"
    return None


def _same_known_task(student_batch, student_idx: int, teacher_batch, teacher_idx: int) -> bool:
    student_task = _task_identity(student_batch, student_idx)
    teacher_task = _task_identity(teacher_batch, teacher_idx)
    return student_task is None or teacher_task is None or student_task == teacher_task


def _decision_get(decision, key: str, default=None):
    if decision is None:
        return default
    if isinstance(decision, Mapping):
        return decision.get(key, default)
    return getattr(decision, key, default)


def _teacher_trajectory_uids(decision, *, include_all: bool = False) -> List[str]:
    key = "teacher_traj_uids" if include_all else "successful_teacher_traj_uids"
    values = _decision_get(decision, key, ()) or ()
    if isinstance(values, str):
        values = (values,)
    ordered = []
    seen = set()
    for value in values:
        traj_uid = str(value)
        if traj_uid not in seen:
            ordered.append(traj_uid)
            seen.add(traj_uid)
    if ordered:
        return ordered
    best = _decision_get(decision, "best_teacher_traj_uid", None)
    return [] if best is None else [str(best)]


def _teacher_aux_gate_active(aux_gate_mode: str, teacher_success: bool, utility: float, utility_eps: float) -> bool:
    if aux_gate_mode == "continuous":
        return True
    if aux_gate_mode == "utility":
        return utility > utility_eps
    if aux_gate_mode == "success_or_utility":
        return teacher_success or utility > utility_eps
    if aux_gate_mode == "success":
        return teacher_success and utility > utility_eps
    raise ValueError(f"Unknown teacher aux_gate_mode: {aux_gate_mode}")


def _trajectory_index(batch) -> Dict[str, Dict[str, List[int]]]:
    """Return group_uid -> traj_uid -> row indices sorted by step_idx."""
    ntb = batch.non_tensor_batch
    group_values = np.asarray(ntb.get("uid", []), dtype=object)
    traj_values = np.asarray(ntb.get("traj_uid", []), dtype=object)
    step_values = np.asarray(ntb.get("step_idx", []), dtype=object)
    n = _num_rows(batch)

    out: Dict[str, Dict[str, List[int]]] = {}
    for i in range(n):
        group_uid = _str_at(group_values, i, f"group-{i}")
        traj_uid = _str_at(traj_values, i, f"traj-{i}")
        out.setdefault(group_uid, {}).setdefault(traj_uid, []).append(i)

    for trajs in out.values():
        for rows in trajs.values():
            rows.sort(key=lambda row: (_int_at(step_values, row, row), row))
    return out


def collect_student_first_cf_span_pairs(
    *,
    student_batch,
    teacher_batch,
    decisions: Mapping[str, object],
    divergence_mode: str = "first",
    rng: Optional[np.random.Generator] = None,
    aux_gate_mode: str = "success",
    utility_eps: float = 0.0,
    max_teacher_pairs_per_student: Optional[int] = 3,
    pair_mode: str = "teacher_better",
    utility_source: str = "reward",
    utility_scale: float = 1.0,
    rescue_pairing: bool = False,
) -> List[StudentFirstCFSpanPair]:
    """Select teacher-vs-student first-divergence CF span candidates.

    ``rescue_pairing``: for each group pair the first successful
    teacher trajectory with the single student sharing the LONGEST common action
    prefix (max t*), emitting exactly one pair per group at that divergence turn.
    Otherwise the legacy value-closeness pairing over all students is used.
    """
    if not decisions or "responses" not in student_batch.batch or "responses" not in teacher_batch.batch:
        return []
    if max_teacher_pairs_per_student is None:
        max_pairs = None
    else:
        max_pairs = max(int(max_teacher_pairs_per_student), 0) or None

    student_index = _trajectory_index(student_batch)
    teacher_index = _trajectory_index(teacher_batch)
    student_steps = np.asarray(student_batch.non_tensor_batch.get("step_idx", []), dtype=object)
    teacher_steps = np.asarray(teacher_batch.non_tensor_batch.get("step_idx", []), dtype=object)
    rng = rng if rng is not None else np.random.default_rng()

    pairs: List[StudentFirstCFSpanPair] = []
    for group_uid, decision in decisions.items():
        group_uid = str(group_uid)
        teacher_success = bool(_decision_get(decision, "teacher_success", False))
        utility = float(_decision_get(decision, "utility", 0.0) or 0.0)
        signed_pairs = pair_mode == "signed"
        if pair_mode not in {"teacher_better", "signed"}:
            raise ValueError(f"Unknown CF pair mode: {pair_mode}")
        if utility_source not in {"reward", "task_score"}:
            raise ValueError(f"Unknown CF utility source: {utility_source}")
        scale = max(abs(float(utility_scale)), 1e-8)
        teacher_traj_uids = _teacher_trajectory_uids(decision, include_all=signed_pairs)
        if not teacher_traj_uids:
            continue
        if not signed_pairs and not _teacher_aux_gate_active(aux_gate_mode, teacher_success, utility, utility_eps):
            continue

        student_trajs = student_index.get(group_uid, {})
        if not student_trajs:
            continue

        teacher_trajectories = []
        for teacher_traj_uid in teacher_traj_uids:
            teacher_rows = teacher_index.get(group_uid, {}).get(teacher_traj_uid, [])
            teacher_reward = _episode_reward(teacher_batch, teacher_rows)
            if not teacher_rows or teacher_reward is None:
                continue
            teacher_task_score = _episode_task_score(teacher_batch, teacher_rows)
            teacher_value = teacher_reward if utility_source == "reward" else teacher_task_score
            if teacher_value is None:
                continue
            teacher_trajectories.append(
                (
                    teacher_traj_uid,
                    teacher_rows,
                    teacher_reward,
                    teacher_task_score,
                    float(teacher_value),
                    [cf_action_at(teacher_batch, row) for row in teacher_rows],
                )
            )
        if not teacher_trajectories:
            continue

        if rescue_pairing:
            # first-success teacher (teacher_traj_uids are already the
            # successful pool, ordered) x student with LONGEST common action prefix.
            (t_uid, t_rows, t_reward, t_task_score, t_value, t_actions) = teacher_trajectories[0]
            best = None  # (t_star, student_traj_uid, student_rows, student_actions, student_value)
            for s_uid, s_rows in student_trajs.items():
                s_reward = _episode_reward(student_batch, s_rows)
                if s_reward is None:
                    continue
                s_task_score = _episode_task_score(student_batch, s_rows)
                s_value = s_reward if utility_source == "reward" else s_task_score
                if s_value is None:
                    continue
                s_actions = [cf_action_at(student_batch, row) for row in s_rows]
                t_star = _first_diverging_turn(t_actions, s_actions)
                if t_star < 0:
                    continue  # student never diverges from a successful teacher
                if best is None or t_star > best[0]:
                    best = (t_star, s_uid, s_rows, s_actions, s_value, s_reward, s_task_score)
            if best is None:
                continue
            t_star, s_uid, s_rows, s_actions, s_value, s_reward, s_task_score = best
            if t_star >= len(t_rows) or t_star >= len(s_rows):
                continue
            teacher_row = int(t_rows[t_star])
            student_row = int(s_rows[t_star])
            if not _same_known_task(student_batch, student_row, teacher_batch, teacher_row):
                continue
            if not _states_align(student_batch, student_row, teacher_batch, teacher_row):
                continue
            pair_utility = (float(t_value) - float(s_value)) / scale
            pairs.append(
                StudentFirstCFSpanPair(
                    pair_id=f"{group_uid}:{t_uid}:{s_uid}:{t_star}",
                    group_uid=group_uid,
                    teacher_traj_uid=t_uid,
                    student_traj_uid=str(s_uid),
                    teacher_row_idx=teacher_row,
                    student_row_idx=student_row,
                    divergence_turn=int(t_star),
                    teacher_step=_int_at(teacher_steps, teacher_row, t_star + 1),
                    student_step=_int_at(student_steps, student_row, t_star + 1),
                    utility=float(pair_utility),
                    teacher_task_id=_task_identity(teacher_batch, teacher_row),
                    student_task_id=_task_identity(student_batch, student_row),
                    teacher_action=t_actions[t_star],
                    student_action=s_actions[t_star],
                    teacher_observation=cf_observation_at(teacher_batch, teacher_row),
                    student_observation=cf_observation_at(student_batch, student_row),
                    teacher_reward=t_reward,
                    student_reward=s_reward,
                    teacher_task_score=t_task_score,
                    student_task_score=s_task_score,
                    group_utility=utility,
                    teacher_response_hash=cf_response_hash_at(teacher_batch, teacher_row),
                    student_response_hash=cf_response_hash_at(student_batch, student_row),
                    suffix_teacher_rows=tuple(int(r) for r in t_rows[t_star + 1:]),
                    critical_divergence_ratio=(float(t_star) / float(len(t_actions))) if t_actions else 0.0,
                    successful_suffix_length=int(max(len(t_rows) - t_star - 1, 0)),
                )
            )
            continue

        for student_traj_uid, student_rows in student_trajs.items():
            student_reward = _episode_reward(student_batch, student_rows)
            if student_reward is None:
                continue
            student_task_score = _episode_task_score(student_batch, student_rows)
            student_value = student_reward if utility_source == "reward" else student_task_score
            if student_value is None:
                continue
            student_actions = [cf_action_at(student_batch, row) for row in student_rows]
            student_pairs = []
            seen_branches = set()
            ordered_teacher_trajectories = sorted(
                teacher_trajectories,
                key=lambda item: (-abs(float(item[4]) - float(student_value)), str(item[0])),
            )
            for (
                teacher_traj_uid,
                teacher_rows,
                teacher_reward,
                teacher_task_score,
                teacher_value,
                teacher_actions,
            ) in ordered_teacher_trajectories:
                pair_utility = (float(teacher_value) - float(student_value)) / scale
                if (not signed_pairs and pair_utility <= 0.0) or (signed_pairs and pair_utility == 0.0):
                    continue
                if divergence_mode == "first":
                    first_k = _first_diverging_turn(teacher_actions, student_actions)
                    k_div_list = [first_k] if first_k >= 0 else []
                elif divergence_mode == "all":
                    k_div_list = _all_diverging_turns(teacher_actions, student_actions)
                elif divergence_mode == "random_one":
                    all_div = _all_diverging_turns(teacher_actions, student_actions)
                    k_div_list = [int(rng.choice(all_div))] if all_div else []
                else:
                    raise ValueError(f"Unknown divergence_mode: {divergence_mode}")

                for k_div in k_div_list:
                    if k_div >= len(teacher_rows) or k_div >= len(student_rows):
                        continue
                    teacher_row = int(teacher_rows[k_div])
                    student_row = int(student_rows[k_div])
                    if not _same_known_task(student_batch, student_row, teacher_batch, teacher_row):
                        continue
                    if not _states_align(student_batch, student_row, teacher_batch, teacher_row):
                        continue
                    teacher_step = _int_at(teacher_steps, teacher_row, k_div + 1)
                    branch_key = (teacher_step, teacher_actions[k_div])
                    if branch_key in seen_branches:
                        continue
                    seen_branches.add(branch_key)
                    student_pairs.append(
                        StudentFirstCFSpanPair(
                            pair_id=f"{group_uid}:{teacher_traj_uid}:{student_traj_uid}:{k_div}",
                            group_uid=group_uid,
                            teacher_traj_uid=teacher_traj_uid,
                            student_traj_uid=str(student_traj_uid),
                            teacher_row_idx=teacher_row,
                            student_row_idx=student_row,
                            divergence_turn=int(k_div),
                            teacher_step=teacher_step,
                            student_step=_int_at(student_steps, student_row, k_div + 1),
                            utility=float(pair_utility),
                            teacher_task_id=_task_identity(teacher_batch, teacher_row),
                            student_task_id=_task_identity(student_batch, student_row),
                            teacher_action=teacher_actions[k_div],
                            student_action=student_actions[k_div],
                            teacher_observation=cf_observation_at(teacher_batch, teacher_row),
                            student_observation=cf_observation_at(student_batch, student_row),
                            teacher_reward=teacher_reward,
                            student_reward=student_reward,
                            teacher_task_score=teacher_task_score,
                            student_task_score=student_task_score,
                            group_utility=utility,
                            teacher_response_hash=cf_response_hash_at(teacher_batch, teacher_row),
                            student_response_hash=cf_response_hash_at(student_batch, student_row),
                        )
                    )
                    if max_pairs is not None and len(student_pairs) >= max_pairs:
                        break
                if max_pairs is not None:
                    if len(student_pairs) >= max_pairs:
                        break
            pairs.extend(student_pairs)

    return pairs
