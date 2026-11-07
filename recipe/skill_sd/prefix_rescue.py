"""Shared-prefix selection for verified skill rescue interventions.

The selector is deliberately independent of Ray and the actor.  It only reads a
student rollout batch and identifies an exact learner state shared by multiple
failed trajectories.  A later stage may query a privileged skill teacher at that
state, but the shared prefix itself never receives auxiliary credit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from recipe.skill_sd.cf_span import cf_action_at, cf_observation_at


@dataclass(frozen=True)
class SharedPrefixCluster:
    group_uid: str
    task_id: Optional[str]
    depth: int
    support: int
    prefix_actions: tuple[str, ...]
    expected_observations: tuple[str, ...]
    state_hash: str
    representative_traj_uid: str
    representative_row: int
    matched_traj_uids: tuple[str, ...]
    matched_rows: tuple[int, ...]
    matched_episode_rewards: tuple[float, ...]
    used_root_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailurePrefixNode:
    """One replayable state in an exact failed-trajectory prefix trie."""

    group_uid: str
    node_uid: str
    parent_uid: Optional[str]
    task_id: Optional[str]
    depth: int
    support: int
    prefix_transitions: tuple[tuple[str, str, str], ...]
    prefix_actions: tuple[str, ...]
    expected_observations: tuple[str, ...]
    representative_traj_uid: str
    representative_row: int
    matched_traj_uids: tuple[str, ...]
    matched_rows: tuple[int, ...]
    matched_episode_rewards: tuple[float, ...]
    is_root: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailurePrefixTrie:
    """The eligible exact-prefix nodes for one failed rollout group."""

    group_uid: str
    task_id: Optional[str]
    root_uid: str
    nodes: tuple[FailurePrefixNode, ...]
    candidate_kind: str = "all_failed"

    def node(self, node_uid: str) -> FailurePrefixNode:
        for node in self.nodes:
            if node.node_uid == str(node_uid):
                return node
        raise KeyError(f"unknown failure-prefix node: {node_uid}")


@dataclass(frozen=True)
class FailurePrefixJob:
    """A self-contained teacher-rescue query at one trie node."""

    group_uid: str
    node_uid: str
    parent_uid: Optional[str]
    task_id: Optional[str]
    depth: int
    support: int
    prefix_transitions: tuple[tuple[str, str, str], ...]
    prefix_actions: tuple[str, ...]
    expected_observations: tuple[str, ...]
    representative_traj_uid: str
    representative_row: int
    matched_traj_uids: tuple[str, ...]
    matched_rows: tuple[int, ...]
    matched_episode_rewards: tuple[float, ...]
    is_root: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _value_at(values, idx: int, default=None):
    if values is None:
        return default
    try:
        return values[idx]
    except Exception:
        return default


def _normalized_text(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _num_rows(batch) -> int:
    responses = getattr(batch, "batch", {}).get("responses")
    if responses is not None:
        return int(responses.shape[0])
    for values in (getattr(batch, "non_tensor_batch", {}) or {}).values():
        try:
            return len(values)
        except TypeError:
            continue
    return 0


def _task_identity(fields: Mapping[str, Any], row: int) -> Optional[str]:
    for key in ("extra.gamefile", "task_id", "session", "seed"):
        value = _value_at(fields.get(key), row, None)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    env_kwargs = _value_at(fields.get("env_kwargs"), row, None)
    if isinstance(env_kwargs, Mapping):
        question = env_kwargs.get("question")
        if question is not None and str(question).strip():
            digest = hashlib.sha1(_normalized_text(question).encode("utf-8")).hexdigest()[:20]
            return f"question_sha1:{digest}"
    return None


def _trajectory_reward(fields: Mapping[str, Any], rows: Sequence[int]) -> Optional[float]:
    values = fields.get("episode_rewards")
    for row in rows:
        value = _value_at(values, row, None)
        try:
            reward = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(reward):
            return reward
    return None


def _next_observation_at(batch, row: int) -> str:
    fields = getattr(batch, "non_tensor_batch", {}) or {}
    return _normalized_text(_value_at(fields.get("next_observation"), row, None))


def _transition_at(batch, row: int) -> Optional[tuple[str, str, str]]:
    observation = _normalized_text(cf_observation_at(batch, row))
    action = _normalized_text(cf_action_at(batch, row))
    next_observation = _next_observation_at(batch, row)
    if not observation or not action or not next_observation:
        return None
    return observation, action, next_observation


def _state_hash(group_uid: str, transitions: Sequence[tuple[str, str, str]], state: str) -> str:
    digest = hashlib.sha1()
    digest.update(str(group_uid).encode("utf-8"))
    for transition in transitions:
        digest.update("\x1f".join(transition).encode("utf-8"))
        digest.update(b"\x1e")
    digest.update(state.encode("utf-8"))
    return digest.hexdigest()[:20]


def _build_cluster(
    *,
    batch,
    group_uid: str,
    task_id: Optional[str],
    trajectories: Mapping[str, Sequence[int]],
    rewards: Mapping[str, float],
    traj_uids: Sequence[str],
    depth: int,
    used_root_fallback: bool,
) -> SharedPrefixCluster:
    ordered_uids = tuple(sorted(str(uid) for uid in traj_uids))
    representative_uid = ordered_uids[0]
    representative_rows = list(trajectories[representative_uid])
    prefix_rows = representative_rows[:depth]
    prefix_transitions = tuple(_transition_at(batch, row) for row in prefix_rows)
    if any(transition is None for transition in prefix_transitions):
        raise ValueError("prefix cluster contains an incomplete transition")
    branch_row = int(representative_rows[depth])
    branch_observation = _normalized_text(cf_observation_at(batch, branch_row))
    expected_observations = tuple(
        [_normalized_text(cf_observation_at(batch, representative_rows[0]))]
        + [_next_observation_at(batch, row) for row in prefix_rows]
    )
    matched_rows = tuple(int(trajectories[uid][depth]) for uid in ordered_uids)
    return SharedPrefixCluster(
        group_uid=str(group_uid),
        task_id=task_id,
        depth=int(depth),
        support=len(ordered_uids),
        prefix_actions=tuple(transition[1] for transition in prefix_transitions),
        expected_observations=expected_observations,
        state_hash=_state_hash(str(group_uid), prefix_transitions, branch_observation),
        representative_traj_uid=representative_uid,
        representative_row=branch_row,
        matched_traj_uids=ordered_uids,
        matched_rows=matched_rows,
        matched_episode_rewards=tuple(float(rewards[uid]) for uid in ordered_uids),
        used_root_fallback=bool(used_root_fallback),
    )


def select_shared_prefix_clusters(
    batch,
    selected_group_uids: Iterable[str],
    *,
    success_threshold: float,
    min_support: int = 2,
    min_depth: int = 1,
    max_depth: int = 10,
    fallback_to_root: bool = True,
    reject_shallow_actions: Sequence[str] = ("look", "inventory"),
    max_reward_range: Optional[float] = 1e-8,
    max_groups: Optional[int] = None,
) -> tuple[list[SharedPrefixCluster], dict[str, float]]:
    """Select one replayable shared-prefix state per all-failed student group.

    Positive-depth candidates use strict transition equality.  The deepest
    candidate wins, with support and a stable trajectory-id tuple as tie breakers.
    The optional depth-0 fallback represents intervention from the common reset
    state and is kept explicitly separate in diagnostics.
    """

    fields = getattr(batch, "non_tensor_batch", {}) or {}
    group_values = np.asarray(fields.get("uid", []), dtype=object)
    traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
    step_values = np.asarray(fields.get("step_idx", []), dtype=object)
    selected = [str(uid) for uid in selected_group_uids]
    if max_groups is not None:
        selected = selected[: max(int(max_groups), 0)]
    selected_set = set(selected)

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in range(_num_rows(batch)):
        group_uid = str(_value_at(group_values, row, f"group-{row}"))
        if group_uid not in selected_set:
            continue
        traj_uid = str(_value_at(traj_values, row, f"traj-{row}"))
        grouped[group_uid][traj_uid].append(row)
    for trajectories in grouped.values():
        for rows in trajectories.values():
            rows.sort(key=lambda row: int(_value_at(step_values, row, row + 1)))

    min_support = max(int(min_support), 2)
    min_depth = max(int(min_depth), 1)
    max_depth = max(int(max_depth), min_depth)
    rejected = {_normalized_text(action) for action in reject_shallow_actions}
    clusters: list[SharedPrefixCluster] = []
    diagnostics = {
        "groups_considered": float(len(selected)),
        "all_failed_groups": 0.0,
        "low_signal_groups": 0.0,
        "nonflat_reward_groups": 0.0,
        "shared_prefix_groups": 0.0,
        "root_fallback_groups": 0.0,
        "no_candidate_groups": 0.0,
        "task_mismatch_groups": 0.0,
        "incomplete_transition_groups": 0.0,
    }

    for group_uid in selected:
        trajectories = grouped.get(group_uid, {})
        if len(trajectories) < min_support:
            diagnostics["no_candidate_groups"] += 1.0
            continue
        rewards = {
            traj_uid: _trajectory_reward(fields, rows)
            for traj_uid, rows in trajectories.items()
        }
        if any(reward is None for reward in rewards.values()) or any(
            float(reward) >= float(success_threshold) for reward in rewards.values()
        ):
            diagnostics["no_candidate_groups"] += 1.0
            continue
        diagnostics["all_failed_groups"] += 1.0
        reward_values = [float(reward) for reward in rewards.values()]
        if max_reward_range is not None and max(reward_values) - min(reward_values) > float(max_reward_range):
            diagnostics["nonflat_reward_groups"] += 1.0
            continue
        diagnostics["low_signal_groups"] += 1.0

        task_ids = {
            identity
            for rows in trajectories.values()
            if rows and (identity := _task_identity(fields, rows[0])) is not None
        }
        if len(task_ids) != 1:
            diagnostics["task_mismatch_groups"] += 1.0
            continue
        task_id = next(iter(task_ids), None)

        transition_cache: dict[str, list[Optional[tuple[str, str, str]]]] = {
            traj_uid: [_transition_at(batch, row) for row in rows]
            for traj_uid, rows in trajectories.items()
        }
        candidates = []
        for depth in range(min_depth, max_depth + 1):
            by_signature: dict[tuple[tuple[str, str, str], ...], list[str]] = defaultdict(list)
            for traj_uid, rows in trajectories.items():
                # A branch action must exist immediately after the shared prefix.
                if len(rows) <= depth:
                    continue
                transitions = transition_cache[traj_uid][:depth]
                if len(transitions) != depth or any(item is None for item in transitions):
                    continue
                signature = tuple(transitions)  # type: ignore[arg-type]
                if depth == 1 and signature[0][1] in rejected:
                    continue
                branch_state = _normalized_text(cf_observation_at(batch, rows[depth]))
                if not branch_state or branch_state != signature[-1][2]:
                    continue
                by_signature[signature].append(traj_uid)
            for signature, members in by_signature.items():
                if len(members) >= min_support:
                    candidates.append((depth, len(members), tuple(sorted(members)), signature))

        if candidates:
            depth, _, members, _ = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            clusters.append(
                _build_cluster(
                    batch=batch,
                    group_uid=group_uid,
                    task_id=task_id,
                    trajectories=trajectories,
                    rewards=rewards,  # type: ignore[arg-type]
                    traj_uids=members,
                    depth=depth,
                    used_root_fallback=False,
                )
            )
            diagnostics["shared_prefix_groups"] += 1.0
            continue

        if fallback_to_root:
            by_initial_state: dict[str, list[str]] = defaultdict(list)
            for traj_uid, rows in trajectories.items():
                if rows:
                    observation = _normalized_text(cf_observation_at(batch, rows[0]))
                    if observation:
                        by_initial_state[observation].append(traj_uid)
            root_members = max(
                (tuple(sorted(members)) for members in by_initial_state.values() if len(members) >= min_support),
                key=lambda members: (len(members), members),
                default=(),
            )
            if root_members:
                clusters.append(
                    _build_cluster(
                        batch=batch,
                        group_uid=group_uid,
                        task_id=task_id,
                        trajectories=trajectories,
                        rewards=rewards,  # type: ignore[arg-type]
                        traj_uids=root_members,
                        depth=0,
                        used_root_fallback=True,
                    )
                )
                diagnostics["root_fallback_groups"] += 1.0
                continue

        if any(any(item is None for item in items) for items in transition_cache.values()):
            diagnostics["incomplete_transition_groups"] += 1.0
        diagnostics["no_candidate_groups"] += 1.0

    if clusters:
        diagnostics["prefix_depth_mean"] = float(np.mean([cluster.depth for cluster in clusters]))
        diagnostics["prefix_depth_max"] = float(max(cluster.depth for cluster in clusters))
        diagnostics["prefix_support_mean"] = float(np.mean([cluster.support for cluster in clusters]))
        diagnostics["matched_trajectories"] = float(sum(cluster.support for cluster in clusters))
    else:
        diagnostics.update({
            "prefix_depth_mean": 0.0,
            "prefix_depth_max": 0.0,
            "prefix_support_mean": 0.0,
            "matched_trajectories": 0.0,
        })
    return clusters, diagnostics


def _valid_transition_chain(
    transitions: Sequence[tuple[str, str, str]],
) -> bool:
    return all(
        transitions[idx - 1][2] == transitions[idx][0]
        for idx in range(1, len(transitions))
    )


def _failure_prefix_node_uid(
    group_uid: str,
    prefix_transitions: Sequence[tuple[str, str, str]],
    state: str,
) -> str:
    return f"fp-{_state_hash(group_uid, prefix_transitions, state)}"


def _node_to_job(node: FailurePrefixNode) -> FailurePrefixJob:
    return FailurePrefixJob(**node.to_dict())


def build_failure_prefix_tries(
    batch,
    selected_group_uids: Iterable[str],
    *,
    success_threshold: float,
    min_support: int = 2,
    max_depth: Optional[int] = None,
    max_reward_range: Optional[float] = 1e-8,
    max_groups: Optional[int] = None,
    group_size: int = 8,
) -> tuple[list[FailurePrefixTrie], dict[str, float]]:
    """Build exact transition tries from failed trajectories.

    Every positive-depth edge is the normalized
    ``(observation, action, next_observation)`` triple. A node is retained only
    when at least ``min_support`` members reach it and each member has a valid
    next-action row. Student next-action diversity is intentionally irrelevant:
    the privileged teacher may still supply the useful divergence. Only complete all-failure groups are eligible.
    """

    fields = getattr(batch, "non_tensor_batch", {}) or {}
    group_values = np.asarray(fields.get("uid", []), dtype=object)
    traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
    step_values = np.asarray(fields.get("step_idx", []), dtype=object)
    selected = list(dict.fromkeys(str(uid) for uid in selected_group_uids))
    if max_groups is not None:
        selected = selected[: max(int(max_groups), 0)]
    selected_set = set(selected)

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in range(_num_rows(batch)):
        group_uid = str(_value_at(group_values, row, f"group-{row}"))
        if group_uid not in selected_set:
            continue
        traj_uid = str(_value_at(traj_values, row, f"traj-{row}"))
        grouped[group_uid][traj_uid].append(row)
    for trajectories in grouped.values():
        for rows in trajectories.values():
            rows.sort(key=lambda row: int(_value_at(step_values, row, row + 1)))

    min_support = max(int(min_support), 2)
    configured_max_depth = None if max_depth is None else int(max_depth)
    diagnostics = {
        "groups_considered": float(len(selected)),
        "all_failed_groups": 0.0,
        "weak_success_candidate_groups": 0.0,
        "weak_success_failed_trajectories": 0.0,
        "weak_success_rejected_groups": 0.0,
        "low_signal_groups": 0.0,
        "nonflat_reward_groups": 0.0,
        "task_mismatch_groups": 0.0,
        "no_root_groups": 0.0,
        "tries_built": 0.0,
        "nodes_built": 0.0,
    }
    tries: list[FailurePrefixTrie] = []

    for group_uid in selected:
        trajectories = grouped.get(group_uid, {})
        if len(trajectories) != group_size:
            diagnostics["no_root_groups"] += 1.0
            continue
        rewards = {
            traj_uid: _trajectory_reward(fields, rows)
            for traj_uid, rows in trajectories.items()
        }
        if any(reward is None for reward in rewards.values()):
            diagnostics["no_root_groups"] += 1.0
            continue

        reward_values = np.asarray(list(rewards.values()), dtype=float)
        if not np.isfinite(reward_values).all() or reward_values.max() >= success_threshold:
            continue
        candidate_kind = "all_failed"
        diagnostics["all_failed_groups"] += 1.0
        if max_reward_range is not None and reward_values.std() > float(max_reward_range):
            diagnostics["nonflat_reward_groups"] += 1.0
            continue
        diagnostics["low_signal_groups"] += 1.0

        task_ids = {
            identity
            for rows in trajectories.values()
            if rows
            and (identity := _task_identity(fields, rows[0])) is not None
        }
        if len(task_ids) != 1:
            diagnostics["task_mismatch_groups"] += 1.0
            continue
        task_id = next(iter(task_ids), None)
        transition_cache = {
            traj_uid: [_transition_at(batch, row) for row in rows]
            for traj_uid, rows in trajectories.items()
        }
        effective_max_depth = (
            max((len(rows) - 1 for rows in trajectories.values()), default=0)
            if configured_max_depth is None or configured_max_depth <= 0
            else configured_max_depth
        )

        by_root_state: dict[str, list[str]] = defaultdict(list)
        for traj_uid, rows in trajectories.items():
            if not rows or not transition_cache[traj_uid]:
                continue
            first_state = _normalized_text(cf_observation_at(batch, rows[0]))
            if first_state and _task_identity(fields, rows[0]) == task_id:
                by_root_state[first_state].append(traj_uid)
        root_members = max(
            (
                tuple(sorted(members))
                for members in by_root_state.values()
                if members
            ),
            key=lambda members: (len(members), members),
            default=(),
        )
        if not root_members:
            diagnostics["no_root_groups"] += 1.0
            continue

        root_state = _normalized_text(cf_observation_at(batch, trajectories[root_members[0]][0]))
        root_uid = _failure_prefix_node_uid(group_uid, (), root_state)
        root_node = FailurePrefixNode(
            group_uid=group_uid,
            node_uid=root_uid,
            parent_uid=None,
            task_id=task_id,
            depth=0,
            support=len(root_members),
            prefix_transitions=(),
            prefix_actions=(),
            expected_observations=(root_state,),
            representative_traj_uid=root_members[0],
            representative_row=int(trajectories[root_members[0]][0]),
            matched_traj_uids=root_members,
            matched_rows=tuple(
                int(trajectories[traj_uid][0]) for traj_uid in root_members
            ),
            matched_episode_rewards=tuple(
                float(rewards[traj_uid]) for traj_uid in root_members
            ),
            is_root=True,
        )
        nodes = [root_node]
        uid_by_signature: dict[
            tuple[tuple[str, str, str], ...], str
        ] = {(): root_uid}

        for depth in range(1, effective_max_depth + 1):
            by_signature: dict[
                tuple[tuple[str, str, str], ...], list[str]
            ] = defaultdict(list)
            for traj_uid in root_members:
                rows = trajectories[traj_uid]
                if len(rows) < depth:
                    continue
                prefix = transition_cache[traj_uid][:depth]
                if (
                    len(prefix) != depth
                    or any(transition is None for transition in prefix)
                ):
                    continue
                signature = tuple(prefix)  # type: ignore[arg-type]
                if not _valid_transition_chain(signature):
                    continue
                by_signature[signature].append(traj_uid)

            for signature, members_list in sorted(
                by_signature.items(), key=lambda item: item[0]
            ):
                members = tuple(sorted(members_list))
                if len(members) < min_support or any(
                    len(trajectories[uid]) <= depth or
                    transition_cache[uid][depth] is None or
                    transition_cache[uid][depth][0] != signature[-1][2] or
                    not _value_at(fields.get("is_action_valid"), trajectories[uid][depth], False)
                    for uid in members
                ):
                    continue
                parent_signature = signature[:-1]
                while parent_signature not in uid_by_signature:
                    parent_signature = parent_signature[:-1]
                parent_uid = uid_by_signature[parent_signature]
                state = signature[-1][2]
                node_uid = _failure_prefix_node_uid(group_uid, signature, state)
                representative_uid = members[0]
                node = FailurePrefixNode(
                    group_uid=group_uid,
                    node_uid=node_uid,
                    parent_uid=parent_uid,
                    task_id=task_id,
                    depth=depth,
                    support=len(members),
                    prefix_transitions=signature,
                    prefix_actions=tuple(item[1] for item in signature),
                    expected_observations=tuple(
                        [signature[0][0]] + [item[2] for item in signature]
                    ),
                    representative_traj_uid=representative_uid,
                    representative_row=int(trajectories[representative_uid][depth]),
                    matched_traj_uids=members,
                    matched_rows=tuple(
                        int(trajectories[traj_uid][depth])
                        for traj_uid in members
                    ),
                    matched_episode_rewards=tuple(
                        float(rewards[traj_uid]) for traj_uid in members
                    ),
                    is_root=False,
                )
                nodes.append(node)
                uid_by_signature[signature] = node_uid

        nodes.sort(
            key=lambda node: (
                node.depth,
                node.prefix_transitions,
                node.node_uid,
            )
        )
        tries.append(
            FailurePrefixTrie(
                group_uid=group_uid,
                task_id=task_id,
                root_uid=root_uid,
                nodes=tuple(nodes),
                candidate_kind=candidate_kind,
            )
        )

    diagnostics["tries_built"] = float(len(tries))
    diagnostics["nodes_built"] = float(
        sum(len(trie.nodes) for trie in tries)
    )
    return tries, diagnostics


def select_deepest_failure_frontier(
    tries: Iterable[FailurePrefixTrie],
    *,
    include_root_if_no_positive: bool = True,
    max_jobs: Optional[int] = None,
) -> list[FailurePrefixJob]:
    """Return all non-nested leaf-most eligible nodes across the tries."""

    jobs = []
    for trie in tries:
        children_by_parent: dict[str, list[FailurePrefixNode]] = defaultdict(list)
        for node in trie.nodes:
            if node.parent_uid is not None:
                children_by_parent[node.parent_uid].append(node)
        positive = [node for node in trie.nodes if not node.is_root]
        frontier = [
            node
            for node in positive
            if not children_by_parent.get(node.node_uid)
        ]
        if not frontier and include_root_if_no_positive:
            frontier = [trie.node(trie.root_uid)]
        jobs.extend(_node_to_job(node) for node in frontier)

    jobs.sort(
        key=lambda job: (
            job.group_uid,
            -job.depth,
            job.prefix_transitions,
            job.node_uid,
        )
    )
    if max_jobs is not None:
        jobs = jobs[: max(int(max_jobs), 0)]
    return jobs


def failure_prefix_backoff_chain(
    trie: FailurePrefixTrie,
    node_uid: str,
    *,
    include_current: bool = True,
) -> list[FailurePrefixJob]:
    """Follow the nearest eligible ancestors from a node through the root."""

    node = trie.node(node_uid)
    chain = [_node_to_job(node)] if include_current else []
    seen = {node.node_uid}
    while node.parent_uid is not None:
        node = trie.node(node.parent_uid)
        if node.node_uid in seen:
            raise ValueError("failure-prefix trie contains a parent cycle")
        seen.add(node.node_uid)
        chain.append(_node_to_job(node))
    return chain


def failure_prefix_job_to_cluster(job: FailurePrefixJob) -> SharedPrefixCluster:
    """Adapt a pure trie job to the trainer's existing cluster contract."""

    return SharedPrefixCluster(
        group_uid=job.group_uid,
        task_id=job.task_id,
        depth=job.depth,
        support=job.support,
        prefix_actions=job.prefix_actions,
        expected_observations=job.expected_observations,
        state_hash=job.node_uid,
        representative_traj_uid=job.representative_traj_uid,
        representative_row=job.representative_row,
        matched_traj_uids=job.matched_traj_uids,
        matched_rows=job.matched_rows,
        matched_episode_rewards=job.matched_episode_rewards,
        used_root_fallback=job.is_root,
    )


def filter_failure_prefix_job_members(
    job: FailurePrefixJob,
    excluded_traj_uids: Iterable[str],
    *,
    min_support: int = 1,
) -> Optional[FailurePrefixJob]:
    """Remove already-consumed members while preserving aligned row metadata."""

    excluded = {str(uid) for uid in excluded_traj_uids}
    kept = [
        idx
        for idx, traj_uid in enumerate(job.matched_traj_uids)
        if str(traj_uid) not in excluded
    ]
    if len(kept) < (1 if job.is_root else max(int(min_support), 1)):
        return None
    representative_idx = kept[0]
    return replace(
        job,
        support=len(kept),
        representative_traj_uid=job.matched_traj_uids[representative_idx],
        representative_row=int(job.matched_rows[representative_idx]),
        matched_traj_uids=tuple(job.matched_traj_uids[idx] for idx in kept),
        matched_rows=tuple(int(job.matched_rows[idx]) for idx in kept),
        matched_episode_rewards=tuple(
            float(job.matched_episode_rewards[idx]) for idx in kept
        ),
    )


def failure_prefix_divergent_job(
    job: FailurePrefixJob,
    batch,
    teacher_action: str,
    *,
    consumed_traj_uids: Iterable[str] = (),
) -> Optional[FailurePrefixJob]:
    """Keep only unconsumed, state-aligned students that differ from teacher."""

    teacher_action = _normalized_text(teacher_action)
    if not teacher_action or not job.expected_observations:
        return None
    expected_state = _normalized_text(job.expected_observations[-1])
    consumed = {str(uid) for uid in consumed_traj_uids}
    excluded = set(consumed)
    for traj_uid, row in zip(job.matched_traj_uids, job.matched_rows):
        student_state = _normalized_text(cf_observation_at(batch, int(row)))
        student_action = _normalized_text(cf_action_at(batch, int(row)))
        if (
            not student_state
            or student_state != expected_state
            or not student_action
            or student_action == teacher_action
        ):
            excluded.add(str(traj_uid))
    return filter_failure_prefix_job_members(
        job, excluded, min_support=1
    )


def failure_prefix_unresolved_backoff_chain(
    trie: FailurePrefixTrie,
    node_uid: str,
    consumed_traj_uids: Iterable[str],
    *,
    include_current: bool = False,
    min_support: int = 2,
) -> list[FailurePrefixJob]:
    """Back off through ancestors after excluding already-trained members."""

    return [
        filtered
        for job in failure_prefix_backoff_chain(
            trie, node_uid, include_current=include_current
        )
        if (
            filtered := filter_failure_prefix_job_members(
                job,
                consumed_traj_uids,
                min_support=min_support,
            )
        ) is not None
    ]


def next_failure_prefix_backoff_job(
    trie: FailurePrefixTrie,
    node_uid: str,
    *,
    consumed_traj_uids: Iterable[str],
    attempted_node_uids: Iterable[str],
    min_support: int = 2,
) -> Optional[FailurePrefixJob]:
    """Return the nearest untried ancestor with enough unresolved members."""

    attempted = {str(uid) for uid in attempted_node_uids}
    for job in failure_prefix_unresolved_backoff_chain(
        trie,
        node_uid,
        consumed_traj_uids,
        include_current=False,
        min_support=min_support,
    ):
        if job.node_uid not in attempted:
            return job
    return None


def split_member_disjoint_failure_jobs(
    jobs: Iterable[FailurePrefixJob],
) -> tuple[list[FailurePrefixJob], list[FailurePrefixJob]]:
    """Select one wave of member-disjoint jobs, deeper nodes first."""

    selected = []
    deferred = []
    active_members: dict[str, list[set[str]]] = defaultdict(list)
    for job in sorted(
        jobs,
        key=lambda item: (
            item.group_uid,
            -int(item.depth),
            -int(item.support),
            item.node_uid,
        ),
    ):
        members = {str(uid) for uid in job.matched_traj_uids}
        if any(
            members.intersection(existing)
            for existing in active_members[job.group_uid]
        ):
            deferred.append(job)
            continue
        selected.append(job)
        active_members[job.group_uid].append(members)
    return selected, deferred


def failure_prefix_completion_advantages(
    job: FailurePrefixJob,
    *,
    scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Return the centered group-completion weights for one verified job."""

    members = tuple(dict.fromkeys(job.matched_traj_uids))
    if not members:
        raise ValueError("failure-prefix job has no matched trajectories")
    root_m = float(np.sqrt(len(members)))
    teacher = float(scale) * root_m
    student = {
        traj_uid: -float(scale) / root_m
        for traj_uid in members
    }
    return teacher, student
