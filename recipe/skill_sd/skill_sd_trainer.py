"""OVCSD training with skill-free GRPO and verified teacher recoveries."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import uuid
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from recipe.skill_sd.cf_span import (
    cf_action_at,
    cf_observation_at,
    cf_response_hash_at,
    collect_student_first_cf_span_pairs,
)
from recipe.skill_sd.teacher_intervention import (
    compute_student_group_stats,
    compute_teacher_acquisition_scores,
    compute_teacher_group_stats,
    compute_teacher_intervention_decisions,
    select_teacher_groups,
    summarize_teacher_intervention,
)
from recipe.skill_sd.teacher_aux_loss import _first_probe_token_indices
from recipe.skill_sd.phi_compiler import shape_rewards_with_phi
from recipe.skill_sd import cf_branch as _cf_branch
from recipe.skill_sd.prefix_rescue import (
    FailurePrefixJob,
    FailurePrefixTrie,
    SharedPrefixCluster,
    build_failure_prefix_tries,
    failure_prefix_backoff_chain,
    failure_prefix_divergent_job,
    failure_prefix_job_to_cluster,
    failure_prefix_unresolved_backoff_chain,
    filter_failure_prefix_job_members,
    next_failure_prefix_backoff_job,
    select_deepest_failure_frontier,
    select_shared_prefix_clusters,
)
from recipe.skill_sd.skill_attribution import (
    SkillUtilityTracker,
    prune_and_weight,
    prune_metrics,
)

PREFIX_RESCUE_STUDENT_ROW_UID = "skill_int_prefix_rescue_student_row_uid"

INTERVENTION_CSV_FIELDS = [
    "global_step",
    "group_uid",
    "student_reward_mean",
    "student_reward_best",
    "student_success_frac",
    "teacher_used",
    "teacher_success",
    "utility",
    "utility_mode",
    "aux_active",
    "teacher_reward_best",
    "teacher_task_score_best",
    "student_task_score_mean",
    "teacher_candidate_groups",
    "teacher_trigger_reward_mean_threshold",
    "active_teacher_groups",
    "G_mean",
    "fixed_G_mean",
    "fixed_with_skill_logp_mean",
    "fixed_no_skill_logp_mean",
    "fixed_G_count",
    "fixed_G_task_count",
    "fixed_G_ready",
    "skill_rows",
    "skill_loss",
    "aux_rows",
]

CPSD_WANDB_METRIC_MAP = {
    "skill_int/student_success_frac": "cpsd/student_success_frac",
    "skill_int/student_all_failed_frac": "cpsd/student_all_failed_frac",
    "skill_int/teacher_candidate_frac": "cpsd/teacher_candidate_frac",
    "skill_int/teacher_query_budget": "cpsd/teacher_query_budget",
    "skill_int/acquisition_score_mean": "cpsd/acquisition_score_mean",
    "skill_int/acquisition_selected_score_mean": "cpsd/acquisition_selected_score_mean",
    "skill_int/teacher_trigger_reward_mean_threshold": "cpsd/teacher_trigger_reward_mean_threshold",
    "skill_int/teacher_success_frac": "cpsd/teacher_success_frac",
    "skill_int/teacher_successful_trajs": "cpsd/teacher_successful_trajs",
    "skill_int/teacher_successful_trajs_per_group": "cpsd/teacher_successful_trajs_per_group",
    "skill_int/state_query_active_groups": "cpsd/state_query_active_groups",
    "skill_int/state_query_selected_groups": "cpsd/state_query_selected_groups",
    "skill_int/state_query_selected_states": "cpsd/state_query_selected_states",
    "skill_int/state_query_generated_rows": "cpsd/state_query_generated_rows",
    "skill_int/state_query_aux_rows": "cpsd/state_query_aux_rows",
    "skill_int/state_query_step_mean": "cpsd/state_query_step_mean",
    "skill_int/state_query_invalid_source_frac": "cpsd/state_query_invalid_source_frac",
    "skill_int/state_query_failed": "cpsd/state_query_failed",
    "skill_int/teacher_task_score_best_mean": "cpsd/teacher_task_score_best_mean",
    "skill_int/teacher_student_task_gap": "cpsd/teacher_student_task_gap",
    "skill_int/U_mean": "cpsd/U_mean",
    "skill_int/U_positive_frac": "cpsd/U_positive_frac",
    "skill_int/active_teacher_frac": "cpsd/active_teacher_frac",
    "skill_int/aux_rows": "cpsd/aux_rows",
    "skill_int/teacher_aux_quality_mean": "cpsd/teacher_aux_quality_mean",
    "skill_int/teacher_aux_quality_positive_frac": "cpsd/teacher_aux_quality_positive_frac",
    "skill_int/int_loss": "cpsd/int_loss",
    "skill_int/int_group_weight_mean": "cpsd/int_group_weight_mean",
    "skill_int/skill_loss": "cpsd/skill_loss",
    "skill_int/skill_rows": "cpsd/skill_rows",
    "skill_int/skill_group_weight_mean": "cpsd/skill_group_weight_mean",
    "skill_int/lambda_skill": "cpsd/lambda_skill",
    "skill_int/effective_lambda_skill": "cpsd/effective_lambda_skill",
    "skill_int/entropy_gate_mean": "cpsd/entropy_gate_mean",
    "skill_int/entropy_mean": "cpsd/entropy_mean",
    "skill_int/fixed_G_mean": "cpsd/fixed_G_mean",
    "skill_int/fixed_with_skill_logp_mean": "cpsd/fixed_with_skill_logp_mean",
    "skill_int/fixed_no_skill_logp_mean": "cpsd/fixed_no_skill_logp_mean",
    "skill_int/fixed_G_count": "cpsd/fixed_G_count",
    "skill_int/fixed_G_task_count": "cpsd/fixed_G_task_count",
    "skill_int/fixed_G_ready": "cpsd/fixed_G_ready",
    "skill_int/cf_loss": "cpsd/cf_loss",
    "skill_int/cf_pairs": "cpsd/cf_pairs",
    "skill_int/cf_delta_mean": "cpsd/cf_delta_mean",
    "skill_int/cf_abs_utility_mean": "cpsd/cf_abs_utility_mean",
    "skill_int/cf_teacher_preferred_frac": "cpsd/cf_teacher_preferred_frac",
    "skill_int/cf_student_preferred_frac": "cpsd/cf_student_preferred_frac",
    "skill_int/cf_gamma_mean": "cpsd/cf_gamma_mean",
    "skill_int/cf_g_first_abs_mean": "cpsd/cf_g_first_abs_mean",
    "skill_int/cf_action_probe_found_frac": "cpsd/cf_action_probe_found_frac",
    "skill_int/cf_probe_action_found_frac": "cpsd/cf_probe_action_found_frac",
    "skill_int/cf_entropy_gate_mean": "cpsd/cf_entropy_gate_mean",
    "skill_int/cf_group_weight_mean": "cpsd/cf_group_weight_mean",
    "skill_int/cf_reward_gap_mean": "cpsd/cf_reward_gap_mean",
    "skill_int/cf_successful_teacher_trajs_used": "cpsd/cf_successful_teacher_trajs_used",
    "skill_int/cf_successful_teacher_traj_use_frac": "cpsd/cf_successful_teacher_traj_use_frac",
    "skill_int/cf_pair_candidates": "cpsd/cf_pair_candidates",
    "skill_int/cf_pair_resolved": "cpsd/cf_pair_resolved",
    "skill_int/cf_pair_drop_missing_teacher": "cpsd/cf_pair_drop_missing_teacher",
    "skill_int/cf_pair_drop_missing_student": "cpsd/cf_pair_drop_missing_student",
    "skill_int/cf_pair_drop_mismatch": "cpsd/cf_pair_drop_mismatch",
    # Counterfactual branch-pair metrics.
    "cf_branch/stall_points": "cpsd/cf_branch_stall_points",
    "cf_branch/confirmed_divergences": "cpsd/cf_branch_confirmed_divergences",
    "cf_branch/replay_mismatch": "cpsd/cf_branch_replay_mismatch",
    "cf_branch/epsilon_cf_dropped": "cpsd/cf_branch_epsilon_cf_dropped",
    "cf_branch/pairs_emitted": "cpsd/cf_branch_pairs_emitted",
    "cf_branch/a_cf_mean": "cpsd/cf_branch_a_cf_mean",
    "cf_branch/teacher_preferred_frac": "cpsd/cf_branch_teacher_preferred_frac",
    "cf_branch/student_preferred_frac": "cpsd/cf_branch_student_preferred_frac",
    "cf_branch/run_to_end_frac": "cpsd/cf_branch_run_to_end_frac",
    "cf_branch/cont_len_mean": "cpsd/cf_branch_cont_len_mean",
    "cf_branch/skill_text_coverage": "cpsd/cf_branch_skill_text_coverage",
    # All-failure groups and verified recovery metrics.
    "skill_int/zero_advantage_failure_group_frac": "cpsd/zero_advantage_failure_group_frac",
    "skill_int/skill_rescue_attempts": "cpsd/skill_rescue_attempts",
    "skill_int/rollouts_until_first_success": "cpsd/rollouts_until_first_success",
    "skill_int/skill_rescue_rate": "cpsd/skill_rescue_rate",
    "skill_int/critical_divergence_step": "cpsd/critical_divergence_step",
    "skill_int/critical_divergence_ratio": "cpsd/critical_divergence_ratio",
    "skill_int/successful_suffix_length": "cpsd/successful_suffix_length",
    "skill_int/extra_rollout_ratio": "cpsd/extra_rollout_ratio",
    # Prefix-conditioned rescue: shared learner state -> verified terminal branch.
    "prefix_rescue/groups_considered": "cpsd/prefix_rescue_groups_considered",
    "prefix_rescue/all_failed_groups": "cpsd/prefix_rescue_all_failed_groups",
    "prefix_rescue/weak_success_candidate_groups": "cpsd/prefix_rescue_weak_success_candidate_groups",
    "prefix_rescue/weak_success_teacher_attempts": "cpsd/prefix_rescue_weak_success_teacher_attempts",
    "prefix_rescue/weak_success_teacher_success_groups": "cpsd/prefix_rescue_weak_success_teacher_success_groups",
    "prefix_rescue/weak_success_pairs_emitted": "cpsd/prefix_rescue_weak_success_pairs_emitted",
    "prefix_rescue/low_signal_groups": "cpsd/prefix_rescue_low_signal_groups",
    "prefix_rescue/nonflat_reward_groups": "cpsd/prefix_rescue_nonflat_reward_groups",
    "prefix_rescue/shared_prefix_groups": "cpsd/prefix_rescue_shared_prefix_groups",
    "prefix_rescue/root_fallback_groups": "cpsd/prefix_rescue_root_fallback_groups",
    "prefix_rescue/no_candidate_groups": "cpsd/prefix_rescue_no_candidate_groups",
    "prefix_rescue/task_mismatch_groups": "cpsd/prefix_rescue_task_mismatch_groups",
    "prefix_rescue/prefix_depth_mean": "cpsd/prefix_rescue_prefix_depth_mean",
    "prefix_rescue/prefix_depth_max": "cpsd/prefix_rescue_prefix_depth_max",
    "prefix_rescue/prefix_support_mean": "cpsd/prefix_rescue_prefix_support_mean",
    "prefix_rescue/matched_trajectories": "cpsd/prefix_rescue_matched_trajectories",
    "prefix_rescue/teacher_attempts": "cpsd/prefix_rescue_teacher_attempts",
    "prefix_rescue/teacher_nonroot_attempts": "cpsd/prefix_rescue_teacher_nonroot_attempts",
    "prefix_rescue/teacher_root_attempts": "cpsd/prefix_rescue_teacher_root_attempts",
    "prefix_rescue/nonroot_attempt_budget": "cpsd/prefix_rescue_nonroot_attempt_budget",
    "prefix_rescue/root_attempt_budget": "cpsd/prefix_rescue_root_attempt_budget",
    "prefix_rescue/max_attempts_per_group": "cpsd/prefix_rescue_max_attempts_per_group",
    "prefix_rescue/configured_attempt_budget_total": "cpsd/prefix_rescue_configured_attempt_budget_total",
    "prefix_rescue/valid_teacher_actions": "cpsd/prefix_rescue_valid_teacher_actions",
    "prefix_rescue/teacher_suffix_generations": "cpsd/prefix_rescue_teacher_suffix_generations",
    "prefix_rescue/teacher_suffix_valid_actions": "cpsd/prefix_rescue_teacher_suffix_valid_actions",
    "prefix_rescue/teacher_suffix_invalid_actions": "cpsd/prefix_rescue_teacher_suffix_invalid_actions",
    "prefix_rescue/teacher_suffix_valid_action_rate": "cpsd/prefix_rescue_teacher_suffix_valid_action_rate",
    "prefix_rescue/teacher_suffix_retries": "cpsd/prefix_rescue_teacher_suffix_retries",
    "prefix_rescue/teacher_suffix_aborted_attempts": "cpsd/prefix_rescue_teacher_suffix_aborted_attempts",
    "prefix_rescue/replay_verified_attempts": "cpsd/prefix_rescue_replay_verified_attempts",
    "prefix_rescue/teacher_success_groups": "cpsd/prefix_rescue_teacher_success_groups",
    "prefix_rescue/teacher_success_rate": "cpsd/prefix_rescue_teacher_success_rate",
    "prefix_rescue/pairs_emitted": "cpsd/prefix_rescue_pairs_emitted",
    "prefix_rescue/matched_student_rows": "cpsd/prefix_rescue_matched_student_rows",
    "prefix_rescue/same_action_rows": "cpsd/prefix_rescue_same_action_rows",
    "prefix_rescue/state_mismatch_rows": "cpsd/prefix_rescue_state_mismatch_rows",
    "prefix_rescue/rollouts_until_first_success": "cpsd/prefix_rescue_rollouts_until_first_success",
    "skill_int/prefix_rescue_attached_pairs": "cpsd/prefix_rescue_attached_pairs",
    # Skill attribution and retrieval metrics.
    "cpsd/skill_util_spread": "cpsd/skill_util_spread",
    "cpsd/skill_util_category_spread": "cpsd/skill_util_category_spread",
    "cpsd/skill_delta_mean": "cpsd/skill_delta_mean",
    "cpsd/skill_delta_spread": "cpsd/skill_delta_spread",
    "cpsd/skill_tracked_count": "cpsd/skill_tracked_count",
    "cpsd/skill_pruned_count": "cpsd/skill_pruned_count",
    "cpsd/skill_dropout_contrast_samples": "cpsd/skill_dropout_contrast_samples",
    "cpsd/skill_kept_weight_mean": "cpsd/skill_kept_weight_mean",
}


@contextmanager
def _disable_retrieval_memory(env_manager) -> Iterator[None]:
    if env_manager is None:
        yield
        return
    saved_mem = getattr(env_manager, "retrieval_memory", None)
    saved_retrieved = getattr(env_manager, "retrieved_memories", None)
    try:
        env_manager.retrieval_memory = None
        env_manager.retrieved_memories = None
        yield
    finally:
        env_manager.retrieval_memory = saved_mem
        env_manager.retrieved_memories = saved_retrieved


@contextmanager
def _force_teacher_alt_prompt_tensors(env_manager, traj_collector, env_num: int) -> Iterator[None]:
    saved_mask = getattr(env_manager, "_skill_mask", None)
    had_force_alt = hasattr(env_manager, "_force_alt_text")
    saved_force_alt = getattr(env_manager, "_force_alt_text", None)
    had_force_init = hasattr(env_manager, "_force_init_skill_prompt")
    saved_force_init = getattr(env_manager, "_force_init_skill_prompt", None)
    had_force_tensors = hasattr(traj_collector, "_force_alt_prompt_tensors")
    saved_force_tensors = getattr(traj_collector, "_force_alt_prompt_tensors", None)
    had_force_lora = hasattr(traj_collector, "_force_teacher_lora_requests")
    saved_force_lora = getattr(traj_collector, "_force_teacher_lora_requests", None)
    try:
        if hasattr(env_manager, "set_skill_mask"):
            env_manager.set_skill_mask(np.ones(int(env_num), dtype=bool))
        else:
            env_manager._skill_mask = np.ones(int(env_num), dtype=bool)
        env_manager._force_alt_text = True
        env_manager._force_init_skill_prompt = True
        traj_collector._force_alt_prompt_tensors = True
        traj_collector._force_teacher_lora_requests = True
        yield
    finally:
        if hasattr(env_manager, "set_skill_mask"):
            env_manager.set_skill_mask(saved_mask)
        else:
            env_manager._skill_mask = saved_mask
        if had_force_alt:
            env_manager._force_alt_text = saved_force_alt
        elif hasattr(env_manager, "_force_alt_text"):
            delattr(env_manager, "_force_alt_text")
        if had_force_init:
            env_manager._force_init_skill_prompt = saved_force_init
        elif hasattr(env_manager, "_force_init_skill_prompt"):
            delattr(env_manager, "_force_init_skill_prompt")
        if had_force_tensors:
            traj_collector._force_alt_prompt_tensors = saved_force_tensors
        elif hasattr(traj_collector, "_force_alt_prompt_tensors"):
            delattr(traj_collector, "_force_alt_prompt_tensors")
        if had_force_lora:
            traj_collector._force_teacher_lora_requests = saved_force_lora
        elif hasattr(traj_collector, "_force_teacher_lora_requests"):
            delattr(traj_collector, "_force_teacher_lora_requests")


@contextmanager
def _force_student_alt_prompt_tensors(env_manager, traj_collector, env_num: int) -> Iterator[None]:
    """Run the student without skills while retaining the with-skill prompt view."""
    saved_mask = getattr(env_manager, "_skill_mask", None)
    had_force_alt = hasattr(env_manager, "_force_alt_text")
    saved_force_alt = getattr(env_manager, "_force_alt_text", None)
    had_force_init = hasattr(env_manager, "_force_init_skill_prompt")
    saved_force_init = getattr(env_manager, "_force_init_skill_prompt", None)
    had_force_tensors = hasattr(traj_collector, "_force_alt_prompt_tensors")
    saved_force_tensors = getattr(traj_collector, "_force_alt_prompt_tensors", None)
    had_force_lora = hasattr(traj_collector, "_force_teacher_lora_requests")
    saved_force_lora = getattr(traj_collector, "_force_teacher_lora_requests", None)
    try:
        no_skill_mask = np.zeros(int(env_num), dtype=bool)
        if hasattr(env_manager, "set_skill_mask"):
            env_manager.set_skill_mask(no_skill_mask)
        else:
            env_manager._skill_mask = no_skill_mask
        env_manager._force_alt_text = True
        env_manager._force_init_skill_prompt = True
        traj_collector._force_alt_prompt_tensors = True
        traj_collector._force_teacher_lora_requests = False
        yield
    finally:
        if hasattr(env_manager, "set_skill_mask"):
            env_manager.set_skill_mask(saved_mask)
        else:
            env_manager._skill_mask = saved_mask
        if had_force_alt:
            env_manager._force_alt_text = saved_force_alt
        elif hasattr(env_manager, "_force_alt_text"):
            delattr(env_manager, "_force_alt_text")
        if had_force_init:
            env_manager._force_init_skill_prompt = saved_force_init
        elif hasattr(env_manager, "_force_init_skill_prompt"):
            delattr(env_manager, "_force_init_skill_prompt")
        if had_force_tensors:
            traj_collector._force_alt_prompt_tensors = saved_force_tensors
        elif hasattr(traj_collector, "_force_alt_prompt_tensors"):
            delattr(traj_collector, "_force_alt_prompt_tensors")
        if had_force_lora:
            traj_collector._force_teacher_lora_requests = saved_force_lora
        elif hasattr(traj_collector, "_force_teacher_lora_requests"):
            delattr(traj_collector, "_force_teacher_lora_requests")


def _prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("val/"):
            out[f"val/{prefix}/{key[len('val/'):]}"] = value
        else:
            out[key] = value
    return out


def _compute_utility(
    teacher_metrics: Dict[str, float],
    student_metrics: Dict[str, float],
    alpha: float = 1.0,
) -> Dict[str, float]:
    util: Dict[str, float] = {}
    for key in teacher_metrics:
        if not key.startswith("val/") or key not in student_metrics:
            continue
        util[f"val/skill_utility/{key[len('val/'):]}"] = alpha * (
            float(teacher_metrics[key]) - float(student_metrics[key])
        )
    return util


def _normalize_causal_probe_mode(value) -> str:
    if isinstance(value, bool):
        return "one_step" if value else "off"
    return str(value)


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


def _summarize_student_intervention(student_stats, selected_group_uids) -> Dict[str, float]:
    if not student_stats:
        return {
            "skill_int/student_groups": 0.0,
            "skill_int/student_trajs": 0.0,
            "skill_int/student_reward_mean": 0.0,
            "skill_int/student_reward_best_mean": 0.0,
            "skill_int/student_success_frac": 0.0,
            "skill_int/student_all_failed_frac": 0.0,
            "skill_int/teacher_candidate_groups": 0.0,
            "skill_int/teacher_candidate_frac": 0.0,
            "skill_int/teacher_candidate_reward_mean": 0.0,
            "skill_int/teacher_candidate_success_frac": 0.0,
        }

    vals = list(student_stats.values())
    selected = [student_stats[g] for g in selected_group_uids if g in student_stats]

    def _mean(items, getter) -> float:
        return float(np.mean([getter(x) for x in items])) if items else 0.0

    return {
        "skill_int/student_groups": float(len(vals)),
        "skill_int/student_trajs": float(sum(s.traj_count for s in vals)),
        "skill_int/student_reward_mean": _mean(vals, lambda s: s.reward_mean),
        "skill_int/student_reward_best_mean": _mean(vals, lambda s: s.reward_best),
        "skill_int/student_success_frac": _mean(vals, lambda s: s.success_frac),
        "skill_int/student_all_failed_frac": _mean(vals, lambda s: float(s.all_failed)),
        "skill_int/teacher_candidate_groups": float(len(selected)),
        "skill_int/teacher_candidate_frac": float(len(selected) / len(vals)),
        "skill_int/teacher_candidate_reward_mean": _mean(selected, lambda s: s.reward_mean),
        "skill_int/teacher_candidate_success_frac": _mean(selected, lambda s: s.success_frac),
        # fraction of ALL groups that are all-fail zero-advantage.
        "skill_int/zero_advantage_failure_group_frac": _mean(vals, lambda s: float(s.all_failed)),
    }


def _infer_env_num(env_manager) -> int:
    candidate_attrs = ("num_envs", "env_num", "num_processes", "batch_size")
    for attr in candidate_attrs:
        n = getattr(env_manager, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    inner = getattr(env_manager, "envs", None)
    for attr in candidate_attrs:
        n = getattr(inner, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    raise RuntimeError("cannot infer env count from env_manager")


class SkillSDTrainer(RayPPOTrainer):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skill_sd_cfg = self.config.get("skill_sd", None)
        self._last_skill_int_teacher_batch = None
        self._last_skill_int_state_query_batch = None
        self._last_skill_int_cf_branch_proto = None
        self._last_skill_int_prefix_rescue_proto = None
        self._last_skill_int_prefix_rescue_suffix_proto = None
        self._skill_int_fixed_ref_ids = set()
        self._skill_int_fixed_ref_order = []
        self._skill_int_fixed_ref_bank = {}
        self._last_fixed_g_record = None
        self._artifact_run_id = uuid.uuid4().hex
        self._artifact_manifest_registered = False
        self._val_dump_dir = self._resolve_val_dump_dir()
        if self._val_dump_dir is not None:
            self._install_val_dump_hook()

        if self.skill_sd_cfg is None:
            print("[SkillSDTrainer] config.skill_sd not set; behaves as vanilla RayPPOTrainer.")
            return
        if self._skill_attribution_enabled():
            self._skill_util_tracker = SkillUtilityTracker()
            self._register_skill_attribution_universe()
        self._load_fixed_g_bank()
        if (
            self._teacher_rollout_enabled()
            or self._cf_branch_enabled()
            or self._process_enabled()
            or self._skill_attribution_enabled()
            or self._prefix_rescue_enabled()
        ):
            self._install_student_first_rollout_hook()
        elif self.skill_sd_cfg.get("teacher_rollout", None) is not None:
            print("[SkillSDTrainer] teacher_rollout disabled; using vanilla student rollout.")

    # ------------------------------------------------------------------ #
    # JSON / dump helpers
    # ------------------------------------------------------------------ #

    def _json_safe(self, value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return self._json_safe(value.item())
            if value.dtype != object and value.size > 1024:
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            return [self._json_safe(v) for v in value.tolist()]
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            tensor = value.detach().cpu()
            if tensor.ndim == 0:
                return self._json_safe(tensor.item())
            if tensor.numel() > 1024:
                return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            return self._json_safe(tensor.numpy())
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _field_at(self, fields: Dict[str, Any], key: str, idx: int, default=None):
        values = fields.get(key)
        if values is None:
            return default
        try:
            return values[idx]
        except Exception:
            return default

    def _int_value(self, value, default: int = 0) -> int:
        try:
            if isinstance(value, np.ndarray):
                value = value.item()
            if isinstance(value, np.generic):
                value = value.item()
            return int(value)
        except Exception:
            return default

    def _float_value(self, value, default: float = 0.0) -> float:
        try:
            if isinstance(value, np.ndarray):
                value = value.item()
            if isinstance(value, np.generic):
                value = value.item()
            return float(value)
        except Exception:
            return default

    def _short_text(self, value, max_chars: int = 320) -> str:
        if value is None:
            return ""
        text = " ".join(str(self._json_safe(value)).split())
        return text if len(text) <= max_chars else text[: max(max_chars - 3, 0)].rstrip() + "..."

    def _default_skill_sd_dir(self, leaf: str) -> Optional[str]:
        config = getattr(self, "config", None)
        trainer_cfg = config.get("trainer", {}) if config is not None else {}
        local_dir = trainer_cfg.get("default_local_dir", None)
        if not local_dir:
            return None
        exp = trainer_cfg.get("experiment_name", "run")
        return os.path.join(local_dir, "skill_sd", str(exp), leaf)

    def _logging_cfg(self):
        if self.skill_sd_cfg is None:
            return {}
        return self.skill_sd_cfg.get("logging", {}) or {}

    def _logging_enabled(self) -> bool:
        return bool(self._logging_cfg().get("enabled", True))

    def _resolve_logging_path(self, key: str, leaf: str) -> Optional[str]:
        if not self._logging_enabled():
            return None
        cfg = self._logging_cfg()
        value = cfg.get(key, None)
        if value:
            return str(value)
        if not bool(cfg.get("default_paths", True)):
            return None
        return self._default_skill_sd_dir(leaf)

    def _append_jsonl(self, path: Optional[str], row: dict) -> None:
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._json_safe(row), ensure_ascii=False) + "\n")

    def _append_artifact(self, record_type: str, payload: dict) -> None:
        """Append one schema-versioned record to this process's gzip archive."""
        artifact_dir = self._default_skill_sd_dir("artifacts")
        if artifact_dir is None:
            return
        os.makedirs(artifact_dir, exist_ok=True)
        if not hasattr(self, "_artifact_run_id"):
            self._artifact_run_id = uuid.uuid4().hex
            self._artifact_manifest_registered = False
        run_id = str(self._artifact_run_id)
        artifact_name = f"records_{run_id}.jsonl.gz"
        artifact_path = os.path.join(artifact_dir, artifact_name)
        if not self._artifact_manifest_registered:
            manifest_path = os.path.join(artifact_dir, "manifest.jsonl.gz")
            manifest = {
                "schema_version": "cpsd.rescue.v1",
                "record_type": "manifest",
                "run_instance_id": run_id,
                "artifact": artifact_name,
                "experiment_name": str(
                    self.config.trainer.get("experiment_name", "run")
                ),
            }
            with gzip.open(manifest_path, "at", encoding="utf-8") as fh:
                fh.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            self._artifact_manifest_registered = True
        envelope = {
            "schema_version": "cpsd.rescue.v1",
            "run_instance_id": run_id,
            "record_type": str(record_type),
            "global_step": int(getattr(self, "global_steps", 0)),
            "payload": self._json_safe(payload),
        }
        with gzip.open(artifact_path, "at", encoding="utf-8") as fh:
            fh.write(json.dumps(envelope, ensure_ascii=False) + "\n")

    def _archive_trajectory_batch(self, batch, *, record_type: str, arm: str) -> None:
        """Archive every trajectory, including failed and otherwise unused rows."""
        try:
            fields = batch.non_tensor_batch
            responses = self.tokenizer.batch_decode(
                batch.batch["responses"], skip_special_tokens=True
            )
            traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
            step_values = np.asarray(fields.get("step_idx", []), dtype=object)
            rows_by_traj: Dict[str, list[int]] = {}
            for row in range(len(batch.batch)):
                traj_uid = str(
                    traj_values[row] if row < len(traj_values) else f"row-{row}"
                )
                rows_by_traj.setdefault(traj_uid, []).append(row)
            for traj_uid, rows in sorted(rows_by_traj.items()):
                rows.sort(
                    key=lambda row: self._int_value(
                        step_values[row] if row < len(step_values) else row + 1,
                        row + 1,
                    )
                )
                first = rows[0]
                self._append_artifact(record_type, {
                    "arm": arm,
                    "uid": self._field_at(fields, "uid", first),
                    "traj_uid": traj_uid,
                    "task_id": self._task_identity(fields, first),
                    "episode_reward": self._field_at(
                        fields, "episode_rewards", first
                    ),
                    "episode_task_score": self._field_at(
                        fields, "episode_task_scores", first
                    ),
                    "skill_text_hash": hashlib.sha1(
                        str(
                            self._field_at(fields, "alt_prompt_text", first, "")
                            or ""
                        ).encode("utf-8")
                    ).hexdigest()[:16],
                    "replay_verified": None,
                    "steps": [
                        {
                            "step": self._int_value(
                                self._field_at(fields, "step_idx", row), row + 1
                            ),
                            "prompt": self._field_at(fields, "prompt_text", row),
                            # Preserve the with-skill prompt used for this state.
                            "alt_prompt": self._field_at(
                                fields, "alt_prompt_text", row
                            ),
                            "observation": self._field_at(fields, "observation", row),
                            "response": responses[row],
                            "model_response": self._field_at(
                                fields, "model_response", row, responses[row]
                            ),
                            "parsed_action": cf_action_at(batch, row),
                            "next_observation": self._field_at(
                                fields, "next_observation", row
                            ),
                            "reward": self._field_at(fields, "rewards", row),
                            "done": self._field_at(fields, "dones", row),
                            "lora_requested": self._field_at(
                                fields, "lora_requested", row, False
                            ),
                        }
                        for row in rows
                    ],
                })
        except Exception as err:
            print(
                f"[SkillSDTrainer] trajectory archive failed (non-fatal): {err}",
                flush=True,
            )

    def _append_csv(self, path: Optional[str], row: dict, fieldnames: Optional[list[str]] = None) -> None:
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        fieldnames = fieldnames or list(row.keys())
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({k: self._json_safe(row.get(k, "")) for k in fieldnames})

    def _resolve_val_dump_dir(self) -> Optional[str]:
        explicit = self.config.trainer.get("val_data_dir", None)
        if explicit:
            return str(explicit)
        local_dir = self.config.trainer.get("default_local_dir", None)
        if not local_dir:
            return None
        exp = self.config.trainer.get("experiment_name", "run")
        return os.path.join(local_dir, "val_trajectories", str(exp))

    def _install_val_dump_hook(self) -> None:
        orig = self._maybe_log_val_generations
        dump_dir = self._val_dump_dir

        def wrapped(inputs, outputs, scores):
            try:
                os.makedirs(dump_dir, exist_ok=True)
                arm = getattr(self, "_val_arm_tag", "default")
                fn = os.path.join(dump_dir, f"step_{self.global_steps:06d}_{arm}_samples.jsonl")
                with open(fn, "a", encoding="utf-8") as fh:
                    for i, (inp, out, score) in enumerate(zip(inputs, outputs, scores)):
                        fh.write(json.dumps(
                            {"idx": i, "input": inp, "output": out, "score": float(score)},
                            ensure_ascii=False,
                        ) + "\n")
                print(f"[SkillSDTrainer] dumped {len(inputs)} val samples -> {fn}", flush=True)
            except Exception as err:
                print(f"[SkillSDTrainer] val sample dump failed (non-fatal): {err}", flush=True)
            return orig(inputs=inputs, outputs=outputs, scores=scores)

        self._maybe_log_val_generations = wrapped

    def _dump_full_val_trajectory_batch(
        self,
        *,
        test_output_gen_batch,
        input_texts,
        val_batch_idx: int,
        example_start: int,
        example_end: int,
        progress_label: str,
    ) -> None:
        arm = getattr(self, "_val_arm_tag", "default")
        if arm == "teacher" and self._fixed_g_enabled() and self._fixed_g_source() == "val_teacher":
            changed = self._register_fixed_g_refs(test_output_gen_batch)
            if changed:
                print(
                    "[SkillSDTrainer] registered fixed-G refs from teacher validation: "
                    f"tasks={len(self._skill_int_fixed_ref_order)}/{self._fixed_g_capacity()}",
                    flush=True,
                )
        if self._val_dump_dir is None:
            return
        try:
            os.makedirs(self._val_dump_dir, exist_ok=True)
            path = os.path.join(self._val_dump_dir, f"step_{self.global_steps:06d}_{arm}_trajectories.jsonl")
            val_repeat_n = max(int(self.config.actor_rollout_ref.rollout.val_kwargs.n), 1)
            configured_val_batch_size = max(int(self.config.data.val_batch_size), 1)
            responses = self.tokenizer.batch_decode(
                test_output_gen_batch.batch["responses"],
                skip_special_tokens=True,
            )
            fields = test_output_gen_batch.non_tensor_batch
            records: Dict[str, Dict[str, Any]] = {}
            order = []
            for row_idx, response in enumerate(responses):
                traj_uid = str(self._json_safe(self._field_at(fields, "traj_uid", row_idx, f"row-{row_idx}")))
                if traj_uid not in records:
                    local_idx = self._int_value(self._field_at(fields, "index", row_idx, len(records)), len(records))
                    batch_local_idx = local_idx // val_repeat_n
                    val_repeat_idx = local_idx % val_repeat_n
                    records[traj_uid] = {
                        "global_step": int(getattr(self, "global_steps", 0)),
                        "arm": arm,
                        "example_index": (val_batch_idx - 1) * configured_val_batch_size + batch_local_idx + 1,
                        "val_batch_index": val_batch_idx,
                        "batch_local_index": batch_local_idx,
                        "rollout_local_index": local_idx,
                        "val_repeat_index": val_repeat_idx,
                        "traj_uid": traj_uid,
                        "uid": self._json_safe(self._field_at(fields, "uid", row_idx)),
                        "task_id": self._json_safe(self._task_identity(fields, row_idx)),
                        "data_source": self._json_safe(self._field_at(fields, "data_source", row_idx)),
                        "input": input_texts[local_idx] if 0 <= local_idx < len(input_texts) else None,
                        "episode_reward": self._json_safe(self._field_at(fields, "episode_rewards", row_idx)),
                        "episode_task_score": self._json_safe(self._field_at(fields, "episode_task_scores", row_idx, 0.0)),
                        "episode_length": self._json_safe(self._field_at(fields, "episode_lengths", row_idx)),
                        "tool_callings": self._json_safe(self._field_at(fields, "tool_callings", row_idx)),
                        "steps": [],
                    }
                    order.append(traj_uid)
                records[traj_uid]["steps"].append({
                    "step": self._int_value(self._field_at(fields, "step_idx", row_idx), len(records[traj_uid]["steps"]) + 1),
                    "prompt": self._json_safe(self._field_at(fields, "prompt_text", row_idx)),
                    "observation": self._json_safe(self._field_at(fields, "observation", row_idx)),
                    "response": response,
                    "model_response": self._json_safe(self._field_at(fields, "model_response", row_idx, response)),
                    "reward": self._json_safe(self._field_at(fields, "rewards", row_idx)),
                    "done": self._json_safe(self._field_at(fields, "dones", row_idx)),
                    "lora_requested": self._json_safe(
                        self._field_at(fields, "lora_requested", row_idx, False)
                    ),
                    "parsed_action": cf_action_at(test_output_gen_batch, row_idx),
                    "replay_verified": None,
                    "active": self._json_safe(self._field_at(fields, "active_masks", row_idx)),
                    "is_action_valid": self._json_safe(self._field_at(fields, "is_action_valid", row_idx)),
                    "next_observation": self._json_safe(self._field_at(fields, "next_observation", row_idx)),
                })
            with open(path, "a", encoding="utf-8") as fh:
                for traj_uid in order:
                    records[traj_uid]["steps"].sort(key=lambda item: item.get("step", 0))
                    fh.write(json.dumps(records[traj_uid], ensure_ascii=False) + "\n")
                    self._append_artifact(
                        "validation_trajectory", records[traj_uid]
                    )
            print(
                f"[SkillSDTrainer] dumped {len(order)} full val trajectories "
                f"({progress_label}, examples {example_start}-{example_end}) -> {path}",
                flush=True,
            )
        except Exception as err:
            print(f"[SkillSDTrainer] full val trajectory dump failed (non-fatal): {err}", flush=True)

    # ------------------------------------------------------------------ #
    # Teacher rollout selection / replay
    # ------------------------------------------------------------------ #

    def init_workers(self):
        cfg = self.skill_sd_cfg
        if (self.config.algorithm.adv_estimator != "grpo" or cfg.skillrl_warmup_steps != 0
                or cfg.student_rollout.use_skill or cfg.prefix_rescue.weak_success.enabled
                or cfg.teacher_rollout.trigger != "hard_only" or cfg.val_arm_mode != "student"
                or not cfg.prefix_rescue.enabled or not cfg.prefix_rescue.replace_teacher_rollout
                or cfg.prefix_rescue.positive_pg_only or cfg.cf_branch.enabled or cfg.process.enabled
                or cfg.skill_attribution.enabled or self.config.env.skills_only_memory.enable_dynamic_update
                or self.config.actor_rollout_ref.actor.optim.lr_warmup_steps != 0
                or self.config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio != 0):
            raise ValueError("This release requires skill-free GRPO and verified all-failure rescue from step one")
        super().init_workers()
        if self.skill_sd_cfg is not None and (
            self._teacher_rollout_enabled()
            or self._cf_branch_enabled()
            or self._process_enabled()
            or self._prefix_rescue_enabled()
        ):
            self._install_student_first_metric_hook()
        if self._cf_branch_enabled() or self._prefix_rescue_enabled():
            self._build_cf_branch_env_pool()

    def _cf_branch_cfg(self):
        if self.skill_sd_cfg is None:
            return {}
        return self.skill_sd_cfg.get("cf_branch", {}) or {}

    def _cf_branch_enabled(self) -> bool:
        return bool(self._cf_branch_cfg().get("enabled", False)) and self._phi_env_name() in ("webshop", "alfworld")

    def _prefix_rescue_cfg(self):
        skill_sd_cfg = getattr(self, "skill_sd_cfg", None)
        if skill_sd_cfg is None:
            return {}
        return skill_sd_cfg.get("prefix_rescue", {}) or {}

    def _prefix_rescue_weak_success_cfg(self):
        return self._prefix_rescue_cfg().get("weak_success", {}) or {}

    def _prefix_rescue_weak_success_enabled(self) -> bool:
        return bool(self._prefix_rescue_weak_success_cfg().get("enabled", False))

    def _prefix_rescue_candidate_group_uids(self, student_stats, selected_group_uids):
        """Only completed task groups with zero student successes enter replay."""
        eligible = {str(uid) for uid, stats in student_stats.items()
                    if stats.traj_count == self.config.env.rollout.n and stats.success_frac == 0}
        return [str(uid) for uid in dict.fromkeys(selected_group_uids) if str(uid) in eligible]

    def _prefix_rescue_enabled(self) -> bool:
        return bool(self._prefix_rescue_cfg().get("enabled", False)) and self._prefix_env_name() in (
            "webshop", "alfworld", "searchqa"
        )

    @staticmethod
    def _rescue_metric_aliases(metrics: dict) -> dict:
        for key, value in list(metrics.items()):
            if key.startswith("prefix_rescue/"):
                metrics.setdefault(f"rescue/{key.split('/', 1)[1]}", value)
        return metrics

    def _print_rescue_job(self, job, record, *, matched_students: int = 0) -> None:
        print(
            "[SkillSDTrainer][RescueJob] "
            f"step={int(getattr(self, 'global_steps', 0))} "
            f"group_uid={job.group_uid} node_uid={job.node_uid} "
            f"attempt={int(record.get('attempt', 0))} "
            f"prefix_depth={int(job.depth)} support={int(job.support)} "
            f"replay_verified={int(bool(record.get('replay_verified', False)))} "
            f"success={int(bool(record.get('success', False)))} "
            f"terminal_reward={float(record.get('terminal_reward', 0.0)):.6f} "
            f"matched_students={int(matched_students)} "
            f"status={record.get('status', 'unknown')}",
            flush=True,
        )

    def _archive_prefix_rescue_attempt(
        self,
        student_batch,
        job,
        record: dict,
    ) -> bool:
        """Immediately persist one executed rescue attempt exactly once."""
        if bool(record.get("_artifact_archived", False)):
            return False

        fields = student_batch.non_tensor_batch
        representative_row = int(job.representative_row)
        skill_text = str(
            record.get("skill_text")
            or _cf_branch.extract_skill_text(
                self._field_at(fields, "alt_prompt_text", representative_row, "")
            )
            or ""
        )
        task_description = str(
            record.get("task_description")
            or self._field_at(fields, "instruction_text", representative_row, "")
            or ""
        )

        rows_by_step = {}
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        step_values = np.asarray(fields.get("step_idx", []), dtype=object)
        for row in range(len(student_batch.batch)):
            if (
                row < len(traj_values)
                and str(traj_values[row]) == str(job.representative_traj_uid)
            ):
                step = self._int_value(
                    step_values[row] if row < len(step_values) else row + 1,
                    row + 1,
                )
                rows_by_step[int(step) - 1] = row

        steps = deepcopy(list(record.get("steps", []) or []))
        for turn, step in enumerate(steps):
            source_row = rows_by_step.get(turn)
            if source_row is not None and turn < int(job.depth):
                step.setdefault(
                    "skill_prompt",
                    self._field_at(fields, "alt_prompt_text", source_row),
                )
                step.setdefault(
                    "no_skill_prompt",
                    self._field_at(fields, "prompt_text", source_row),
                )
                step["generation_mode"] = "prefix_replay"
            elif turn == int(job.depth):
                step.setdefault("skill_prompt", record.get("teacher_query_prompt"))
                step.setdefault(
                    "no_skill_prompt", record.get("teacher_query_no_skill_prompt")
                )
                step["generation_mode"] = "teacher_intervention"
            else:
                step.setdefault("skill_prompt", None)
                step.setdefault("no_skill_prompt", None)
                step["generation_mode"] = "teacher_suffix"
            step["skill_text_hash"] = record.get("skill_text_hash")

        expected_observations = list(job.expected_observations or ())
        intervention_observation = (
            expected_observations[int(job.depth)]
            if int(job.depth) < len(expected_observations)
            else None
        )
        payload = {
            "attempt_key": (
                f"{job.group_uid}:{job.node_uid}:"
                f"{int(record.get('attempt', 0))}"
            ),
            "group_uid": str(job.group_uid),
            "task_id": str(job.task_id),
            "representative_traj_uid": str(job.representative_traj_uid),
            "representative_row": representative_row,
            "matched_traj_uids": list(job.matched_traj_uids),
            "matched_rows": list(job.matched_rows),
            "matched_students": int(record.get("matched_students", 0)),
            "prefix_actions": list(job.prefix_actions),
            "expected_observations": expected_observations,
            "intervention_depth": int(job.depth),
            "intervention_turn": int(job.depth),
            "intervention_observation": intervention_observation,
            "task_description": task_description,
            "skill_text": skill_text,
            "skill_text_hash": record.get("skill_text_hash"),
            "teacher_skill_prompt": record.get("teacher_query_prompt"),
            "teacher_no_skill_prompt": record.get(
                "teacher_query_no_skill_prompt"
            ),
            "teacher_raw_response": record.get("teacher_query_raw_response"),
            "teacher_action": record.get("teacher_action"),
            "failure_reason": (
                None
                if bool(record.get("success"))
                and bool(record.get("signal_emitted"))
                else record.get("status", "unknown")
            ),
            "steps": steps,
            **{
                key: value
                for key, value in record.items()
                if key not in {"steps", "skill_text", "_artifact_archived"}
            },
        }
        self._append_artifact("teacher_rescue_attempt", payload)
        record["_artifact_archived"] = True
        return True

    def _branch_replay_pool_size(self) -> int:
        sizes = []
        if self._cf_branch_enabled():
            sizes.append(int(self._cf_branch_cfg().get("replay_batch_workers", 24)))
        if self._prefix_rescue_enabled():
            sizes.append(int(self._prefix_rescue_cfg().get("replay_batch_workers", 24)))
        # Lightweight unit-test trainers override only _cf_branch_cfg and do not
        # construct the full config object.
        if not sizes:
            sizes.append(int(self._cf_branch_cfg().get("replay_batch_workers", 24)))
        return max([1, *sizes])

    def _skill_attribution_cfg(self):
        if self.skill_sd_cfg is None:
            return {}
        return self.skill_sd_cfg.get("skill_attribution", {}) or {}

    def _skill_attribution_enabled(self) -> bool:
        return bool(self._skill_attribution_cfg().get("enabled", False))

    def _register_skill_attribution_universe(self) -> None:
        if not self._skill_attribution_enabled():
            return
        tracker = getattr(self, "_skill_util_tracker", None)
        memory = getattr(getattr(self, "envs", None), "retrieval_memory", None)
        skill_ids = []
        for skill in (getattr(memory, "general_skills", None) or []):
            if skill.get("skill_id") is not None:
                skill_ids.append(str(skill["skill_id"]))
        for category_skills in (getattr(memory, "task_specific_skills", None) or {}).values():
            for skill in category_skills or []:
                if skill.get("skill_id") is not None:
                    skill_ids.append(str(skill["skill_id"]))
        if tracker is not None:
            tracker.register_universe(skill_ids)

    def _skill_attribution_task_type(self, task_description: str) -> str:
        if not bool(self._skill_attribution_cfg().get("dedust_by_task_type", True)):
            return "__all__"
        memory = getattr(getattr(self, "envs", None), "retrieval_memory", None)
        detector = getattr(memory, "_detect_task_type", None)
        if callable(detector):
            try:
                return str(detector(str(task_description or "")))
            except Exception:
                pass
        return "unknown"

    @staticmethod
    def _normalize_skill_ids(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        try:
            return [str(skill_id) for skill_id in value if skill_id is not None]
        except TypeError:
            return [str(value)]

    def _refresh_skill_attribution_feedback(self) -> Dict[str, float]:
        if not self._skill_attribution_enabled():
            return {}
        tracker = getattr(self, "_skill_util_tracker", None)
        if tracker is None:
            tracker = SkillUtilityTracker()
            self._skill_util_tracker = tracker
            self._register_skill_attribution_universe()

        cfg = self._skill_attribution_cfg()
        pruned, weights = prune_and_weight(
            tracker,
            prune_threshold=float(cfg.get("prune_threshold", 0.0)),
            min_samples=int(cfg.get("min_samples_per_skill", 10)),
            weight_by_delta=bool(cfg.get("weight_by_delta", True)),
        )
        memory = getattr(getattr(self, "envs", None), "retrieval_memory", None)
        if memory is not None:
            if hasattr(memory, "set_skill_weights"):
                memory.set_skill_weights(weights)
            else:
                memory.skill_weights = dict(weights)

        contrast_samples = sum(
            tracker.contrast_count(skill_id)
            for skill_id in tracker.all_skill_ids()
        )
        metrics = tracker.summary()
        metrics.update(prune_metrics(pruned, weights, contrast_samples))
        return metrics

    def _observe_skill_attribution(
        self,
        pairs,
        student_batch,
        branch_results,
    ) -> Dict[str, float]:
        if not self._skill_attribution_enabled():
            return {}
        tracker = getattr(self, "_skill_util_tracker", None)
        if tracker is None:
            tracker = SkillUtilityTracker()
            self._skill_util_tracker = tracker
        # Dynamic skills may be added after trainer construction.
        self._register_skill_attribution_universe()

        fields = student_batch.non_tensor_batch
        skill_ids_col = fields.get("skill_ids")
        task_type_by_row = {
            int(result["student_row"]): str(result.get("task_type", "unknown"))
            for result in branch_results
            if result.get("student_row") is not None
        }
        for pair in pairs:
            row = int(pair.student_row)
            skill_ids = (
                self._normalize_skill_ids(skill_ids_col[row])
                if skill_ids_col is not None and row < len(skill_ids_col)
                else []
            )
            task_type = task_type_by_row.get(row)
            if task_type is None:
                task_desc = self._field_at(fields, "instruction_text", row, "")
                task_type = self._skill_attribution_task_type(str(task_desc or ""))
            tracker.observe_decision(
                a_cf=float(pair.a_cf),
                task_type=task_type,
                skill_ids=skill_ids,
            )
        return self._refresh_skill_attribution_feedback()

    def _build_cf_branch_env_pool(self) -> None:
        """Create a separate environment pool for prefix replay. Task keys are: WebShop (session=int goal),
        ALFWorld (session=gamefile path), or SearchQA (reset kwargs)."""
        self._cf_branch_envs = None
        self._cf_branch_env = self._prefix_env_name() if self._prefix_rescue_enabled() else self._phi_env_name()
        pool_n = self._branch_replay_pool_size()
        resources = {"num_cpus": float(getattr(self.config.env.resources_per_worker, "num_cpus", 0.2))}
        try:
            import os as _os
            if self._cf_branch_env == "alfworld":
                from agent_system.environments.env_package.alfworld import build_alfworld_envs
                alf_config_path = _os.path.join(
                    _os.path.dirname(_os.path.abspath(__file__)), "..", "..",
                    "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")
                env_kwargs = {"eval_dataset": getattr(getattr(self.config.env, "alfworld", {}),
                                                      "eval_dataset", "eval_in_distribution")}
                self._cf_branch_envs = build_alfworld_envs(
                    alf_config_path=alf_config_path, seed=int(self.config.env.seed) + 2000,
                    env_num=pool_n, group_n=1, is_train=True, env_kwargs=env_kwargs,
                    resources_per_worker=resources)
            elif self._cf_branch_env == "searchqa":
                from agent_system.environments.env_package.search import build_search_envs
                self._cf_branch_envs = build_search_envs(
                    seed=int(self.config.env.seed) + 2000,
                    env_num=pool_n,
                    group_n=1,
                    is_train=True,
                    env_config=self.config.env,
                )
            else:
                from agent_system.environments.env_package.webshop import build_webshop_envs
                use_small = bool(getattr(self.config.env, "webshop", {}).get("use_small", True)) \
                    if hasattr(self.config.env, "webshop") else True
                base = "agent_system/environments/env_package/webshop/webshop/data"
                root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
                fp = _os.path.join(root, base, "items_shuffle_1000.json" if use_small else "items_shuffle.json")
                ap = _os.path.join(root, base, "items_ins_v2_1000.json" if use_small else "items_ins_v2.json")
                env_kwargs = {"observation_mode": "text", "num_products": None, "human_goals": False,
                              "file_path": fp, "attr_path": ap}
                self._cf_branch_envs = build_webshop_envs(
                    seed=int(self.config.env.seed) + 2000, env_num=pool_n, group_n=1,
                    resources_per_worker=resources, is_train=True, env_kwargs=env_kwargs)
            print(f"[SkillSDTrainer] cf_branch isolated env pool ready "
                  f"(env={self._cf_branch_env}, workers={pool_n}).", flush=True)
        except Exception as err:
            print(f"[SkillSDTrainer] cf_branch env pool build failed: {err}", flush=True)
            raise RuntimeError("Could not initialize the prefix replay environment pool") from err

    # ----- branch env adapters (normalize WebShop / ALFWorld / SearchQA) ------- #
    def _cf_reset_pool(self, sessions):
        """Reset branch slots to task identities and return (obs, info) lists."""
        raw = self._cf_branch_envs
        env = getattr(self, "_cf_branch_env", "webshop")
        if env == "alfworld":
            text_obs, _img, info_list = raw.reset(gamefiles=[str(s) for s in sessions])
            return text_obs, info_list
        if env == "searchqa":
            return raw.reset(kwargs=[deepcopy(session) for session in sessions])
        return raw.reset(sessions=sessions, repeat_sessions=False)

    def _cf_step_pool(self, actions):
        """Step the branch pool. Returns (obs, rewards, dones, infos) uniformly
        (ALFWorld's 5-tuple drops image_obs; WebShop's 4-tuple passes through)."""
        raw = self._cf_branch_envs
        if getattr(self, "_cf_branch_env", "webshop") == "alfworld":
            text_obs, _img, rr, dd, ii = raw.step(actions)
            return text_obs, rr, dd, ii
        return raw.step(actions)

    def _cf_extract_obs_avail(self, info):
        """Pull (observation_text, available_actions) from a branch-pool info dict.
        WebShop: observation + available_actions.clickables; ALFWorld:
        observation_text + admissible_commands."""
        info = info if isinstance(info, dict) else {}
        if getattr(self, "_cf_branch_env", "webshop") == "alfworld":
            return info.get("observation_text"), (info.get("admissible_commands", []) or [])
        if getattr(self, "_cf_branch_env", "webshop") == "searchqa":
            return None, []
        return info.get("observation"), (info.get("available_actions", {}) or {}).get("clickables", [])

    def _cf_placeholder_action(self) -> str:
        """Masked-slot placeholder action (result ignored). ALFWorld tolerates any
        inadmissible string ('look'); WebShop uses 'click[back to search]'."""
        env = getattr(self, "_cf_branch_env", "webshop")
        if env == "alfworld":
            return "look"
        if env == "searchqa":
            return "<answer></answer>"
        return "click[back to search]"

    def _cf_compile_milestone(self, spec):
        """Compile Φ milestones for a branch spec. ALFWorld keys off the gamefile
        (spec['session']); WebShop off the task description."""
        from recipe.skill_sd import phi_compiler as _phi
        if getattr(self, "_cf_branch_env", "webshop") == "alfworld":
            return _phi.compile_milestones("alfworld", gamefile=str(spec.get("session", "") or ""))
        return _phi.compile_milestones("webshop", task_desc=str(spec.get("task_desc", "") or ""))

    def _teacher_rollout_cfg(self):
        if self.skill_sd_cfg is None:
            return None
        return self.skill_sd_cfg.get("teacher_rollout", None)

    def _teacher_rollout_enabled(self) -> bool:
        cfg = self._teacher_rollout_cfg()
        return cfg is not None and bool(cfg.get("enabled", False))

    def _state_teacher_query_cfg(self):
        if self.skill_sd_cfg is None:
            return {}
        return self.skill_sd_cfg.get("state_teacher_query", {}) or {}

    def _state_teacher_query_enabled(self) -> bool:
        cfg = self._state_teacher_query_cfg()
        return bool(cfg.get("enabled", False)) and float(self.skill_sd_cfg.get("lambda_int", 0.0)) > 0.0

    def _curriculum_teacher_n(self, cfg) -> int:
        static_n = max(int(cfg.get("n", 1)), 1)
        n_start = cfg.get("n_start", None)
        n_end = cfg.get("n_end", None)
        if n_start is None or n_end is None:
            return static_n
        n_start = max(int(n_start), 1)
        n_end = max(int(n_end), 1)
        steps = max(int(cfg.get("n_curriculum_steps", 30)), 1)
        t = min(max(int(getattr(self, "global_steps", 0)), 0), steps) / steps
        value = n_start + (n_end - n_start) * t
        lo, hi = sorted((n_start, n_end))
        return int(min(max(round(value), lo), hi))

    @staticmethod
    def _optional_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.lower() in {"", "none", "null"}:
            return None
        return float(value)

    def _curriculum_trigger_reward_mean_threshold(self, cfg, success_threshold: float) -> float:
        start = self._optional_float(cfg.get("trigger_reward_mean_start", None))
        end = self._optional_float(cfg.get("trigger_reward_mean_end", None))
        if start is not None and end is not None:
            steps = max(int(cfg.get("trigger_reward_mean_curriculum_steps", 30)), 1)
            t = min(max(int(getattr(self, "global_steps", 0)), 0), steps) / steps
            lo, hi = sorted((start, end))
            return float(min(max(start + (end - start) * t, lo), hi))

        static = self._optional_float(cfg.get("trigger_reward_mean_threshold", None))
        if static is not None:
            return float(static)
        return float(success_threshold)

    def _teacher_intervention_rng(self):
        rng = getattr(self, "_skill_int_rng", None)
        if rng is None:
            seed = int(getattr(self.config, "env", {}).get("seed", 0))
            rng = np.random.default_rng(seed)
            self._skill_int_rng = rng
        return rng

    def _compute_student_intervention_stats(self, batch):
        cfg = self._teacher_rollout_cfg()
        if cfg is None:
            return {}, [], {}
        success_threshold = float(cfg.get("success_reward_threshold", 10.0))
        candidate_success_threshold = self._optional_float(
            cfg.get('candidate_success_reward_threshold', None)
        )
        if candidate_success_threshold is None:
            candidate_success_threshold = success_threshold
        trigger = str(cfg.get("trigger", "hard_only"))
        trigger_probability = float(cfg.get("trigger_probability", 0.0))
        query_budget = cfg.get("query_budget", None)
        query_budget = None if query_budget is None else int(query_budget)
        acquisition_signal = str(cfg.get("acquisition_signal", "task_score"))
        acquisition_floor = float(cfg.get("acquisition_floor", 0.05))
        acquisition_hardness_power = float(cfg.get("acquisition_hardness_power", 1.0))
        acquisition_uncertainty_weight = float(cfg.get("acquisition_uncertainty_weight", 0.25))
        teacher_n = self._curriculum_teacher_n(cfg)
        trigger_reward_mean_threshold = self._curriculum_trigger_reward_mean_threshold(
            cfg, candidate_success_threshold
        )
        selection_threshold = (
            trigger_reward_mean_threshold
            if trigger == 'mean_below'
            else candidate_success_threshold
        )

        student_stats = compute_student_group_stats(
            batch, success_reward_threshold=candidate_success_threshold
        )
        selected_group_uids = select_teacher_groups(
            student_stats,
            trigger=trigger,
            success_reward_threshold=selection_threshold,
            probability=trigger_probability,
            budget=query_budget,
            acquisition_signal=acquisition_signal,
            acquisition_floor=acquisition_floor,
            acquisition_hardness_power=acquisition_hardness_power,
            acquisition_uncertainty_weight=acquisition_uncertainty_weight,
            rng=self._teacher_intervention_rng(),
        )
        if (
            self._prefix_rescue_weak_success_enabled()
            and self._prefix_rescue_enabled()
        ):
            selected_group_uids = self._prefix_rescue_candidate_group_uids(
                student_stats, selected_group_uids
            )
        acquisition_scores = compute_teacher_acquisition_scores(
            student_stats,
            signal=acquisition_signal,
            reward_scale=candidate_success_threshold,
            floor=acquisition_floor,
            hardness_power=acquisition_hardness_power,
            uncertainty_weight=acquisition_uncertainty_weight,
        )
        metrics = _summarize_student_intervention(student_stats, selected_group_uids)
        metrics.update({
            "skill_int/teacher_rollout_n": float(teacher_n),
            "skill_int/teacher_trigger_probability": float(trigger_probability),
            "skill_int/success_reward_threshold": success_threshold,
            'skill_int/candidate_success_reward_threshold': float(candidate_success_threshold),
            "skill_int/teacher_trigger_reward_mean_threshold": float(trigger_reward_mean_threshold),
            "skill_int/teacher_query_budget": float(query_budget if query_budget is not None else len(student_stats)),
            "skill_int/acquisition_score_mean": float(np.mean(list(acquisition_scores.values()))) if acquisition_scores else 0.0,
            "skill_int/acquisition_selected_score_mean": (
                float(np.mean([acquisition_scores[str(uid)] for uid in selected_group_uids]))
                if selected_group_uids else 0.0
            ),
        })
        return metrics, selected_group_uids, student_stats

    def _state_query_row_confidence(self, student_batch, row: int) -> float:
        rollout_log_probs = student_batch.batch.get("rollout_log_probs")
        if rollout_log_probs is None:
            return 0.0
        response_len = int(student_batch.batch["responses"].shape[-1])
        if "loss_mask" in student_batch.batch.keys():
            response_mask = student_batch.batch["loss_mask"][row, -response_len:]
        else:
            response_mask = student_batch.batch["attention_mask"][row, -response_len:]
        mask = response_mask.to(rollout_log_probs.dtype)
        denom = mask.sum().clamp(min=1.0)
        return float(((rollout_log_probs[row] * mask).sum() / denom).detach().cpu())

    def _select_student_state_query_records(
        self,
        student_batch,
        group_utilities: Dict[str, float],
        group_qualities: Optional[Dict[str, float]] = None,
        force_selection: Optional[str] = None,
    ) -> list[dict]:
        cfg = self._state_teacher_query_cfg()
        # cf_branch reuses this selection independently of the
        # state_teacher_query switch, with force_selection="phi_stall".
        if not (self._state_teacher_query_enabled() or (force_selection is not None and self._cf_branch_enabled())) or not group_utilities:
            return []
        required = ("alt_input_ids", "alt_attention_mask")
        if any(key not in student_batch.batch.keys() for key in required):
            return []

        max_states = max(int(cfg.get("max_states_per_group", 2)), 0)
        max_groups = max(int(cfg.get("max_groups_per_step", 8)), 0)
        min_step = max(int(cfg.get("min_step", 2)), 1)
        failed_only = bool(cfg.get("failed_only", True))
        prefer_invalid = bool(cfg.get("prefer_invalid", True))
        selection = force_selection or str(cfg.get("selection", "low_confidence"))
        success_threshold = float((self._teacher_rollout_cfg() or {}).get("success_reward_threshold", 10.0))
        if max_states <= 0 or max_groups <= 0:
            return []

        fields = student_batch.non_tensor_batch
        group_values = np.asarray(fields.get("uid", []), dtype=object)
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        grouped: Dict[str, Dict[str, list[int]]] = {}
        for row, group_uid_raw in enumerate(group_values):
            group_uid = str(group_uid_raw)
            if group_uid not in group_utilities:
                continue
            traj_uid = str(traj_values[row]) if row < len(traj_values) else f"row:{row}"
            grouped.setdefault(group_uid, {}).setdefault(traj_uid, []).append(row)

        ordered_groups = sorted(group_utilities, key=lambda uid: (-float(group_utilities[uid]), str(uid)))[:max_groups]
        records = []
        for group_uid in ordered_groups:
            trajectories = grouped.get(group_uid, {})
            eligible = []
            for traj_uid, rows in trajectories.items():
                rows = sorted(rows, key=lambda idx: self._int_value(self._field_at(fields, "step_idx", idx), idx + 1))
                reward = self._float_value(self._field_at(fields, "episode_rewards", rows[0], 0.0))
                if failed_only and reward >= success_threshold:
                    continue
                eligible.append((reward, traj_uid, rows))
            if not eligible:
                continue

            # The best failed trajectory is closest to useful behavior while still
            # visiting states where the student needs expert correction.
            reward, traj_uid, rows = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
            candidates = []
            # Φ-stall selection precomputes ΔΦ per row on this
            # trajectory (rows already ordered by step). Stall = ΔΦ≤0.
            stall_map: Dict[int, float] = {}
            if selection == "phi_stall":
                stall_map = self._phi_stall_delta_map(student_batch, rows)
            for row in rows:
                step = self._int_value(self._field_at(fields, "step_idx", row), 0)
                if step < min_step:
                    continue
                is_valid = bool(self._field_at(fields, "is_action_valid", row, True))
                confidence = self._state_query_row_confidence(student_batch, row)
                invalid_rank = 0 if prefer_invalid and not is_valid else 1
                if selection == "latest":
                    rank = (invalid_rank, -step, row)
                    reason = "invalid_action" if not is_valid else "latest"
                elif selection == "earliest":
                    rank = (invalid_rank, step, row)
                    reason = "invalid_action" if not is_valid else "earliest"
                elif selection == "low_confidence":
                    rank = (invalid_rank, confidence, step, row)
                    reason = "invalid_action" if not is_valid else "low_confidence"
                elif selection == "phi_stall":
                    # Most-negative ΔΦ (most stuck) first.
                    dphi = stall_map.get(row, 0.0)
                    rank = (invalid_rank, dphi, step, row)
                    reason = "phi_stall"
                else:
                    raise ValueError(f"Unknown skill_sd.state_teacher_query.selection: {selection}")
                candidates.append((rank, row, step, is_valid, confidence, reason))

            for _, row, step, is_valid, confidence, reason in sorted(candidates)[:max_states]:
                records.append({
                    "group_uid": group_uid,
                    "utility": float(group_utilities[group_uid]),
                    "quality": float((group_qualities or {}).get(group_uid, 1.0)),
                    "student_row": int(row),
                    "student_traj_uid": traj_uid,
                    "student_step": int(step),
                    "student_episode_reward": float(reward),
                    "student_action_valid": bool(is_valid),
                    "student_action_logp_mean": float(confidence),
                    "selection_reason": reason,
                })
        return records

    def _alfworld_gamefile(self, fields, idx: int) -> str:
        """ALFWorld gamefile PATH for row ``idx`` (phi_compiler parses task type +
        object/receptacle from the path). Reads ``extra.gamefile`` first — ``task_id``
        is a trajectory id like 'trial_00003_...' that the milestone regex can't
        parse, so falling back to it yields degenerate (obj=None) milestones."""
        return str(
            self._field_at(fields, "extra.gamefile", idx, "")
            or self._field_at(fields, "task_id", idx, "")
            or self._field_at(fields, "uid", idx, "")
        )

    @staticmethod
    def _alfworld_task_from_text(text: Any) -> str:
        value = str(text or "")
        marker = "your task is to:"
        start = value.lower().find(marker)
        if start < 0:
            return ""
        tail = value[start + len(marker):].strip()
        return tail.splitlines()[0].strip() if tail else ""

    def _alfworld_task_description(self, fields, idx: int, fallback_text: Any = "") -> str:
        explicit = str(self._field_at(fields, "instruction_text", idx, "") or "").strip()
        if explicit:
            return explicit
        for key in ("prompt_text", "alt_prompt_text", "observation"):
            task = self._alfworld_task_from_text(self._field_at(fields, key, idx, ""))
            if task:
                return task
        return self._alfworld_task_from_text(fallback_text)

    def _phi_stall_delta_map(self, student_batch, rows) -> Dict[int, float]:
        """ΔΦ per row for one trajectory (rows ordered by step). Used by Φ-stall
        selection. Returns {row: ΔΦ_t} for the transition out of
        that row; stall = ΔΦ ≤ 0."""
        env = self._phi_env_name()
        if env is None or len(rows) < 2:
            return {}
        from recipe.skill_sd import phi_compiler as _phi
        fields = student_batch.non_tensor_batch
        obs_seq = [str(self._field_at(fields, "observation", r, "") or "") for r in rows]
        act_seq = [str(self._field_at(fields, "model_response", r, "") or "") for r in rows]
        if env == "webshop":
            task = str(self._field_at(fields, "instruction_text", rows[0], "") or self._field_at(fields, "session", rows[0], ""))
            milestones = _phi.compile_milestones("webshop", task_desc=task)
        else:
            task = self._alfworld_gamefile(fields, rows[0])
            milestones = _phi.compile_milestones("alfworld", gamefile=task)
        term_r = self._float_value(self._field_at(fields, "episode_rewards", rows[0], 0.0))
        curve = _phi.phi_curve(env, milestones, obs_seq, act_seq, terminal_reward=term_r)
        gamma = float(self._process_cfg().get("gamma", 0.95))
        dphi = _phi.delta_phi(curve, gamma=gamma)  # len rows-1, index t -> rows[t]
        return {int(rows[t]): float(dphi[t]) for t in range(len(dphi))}

    def _select_first_traj_rows_by_group(self, batch, selected_group_uids):
        selected = set(str(g) for g in selected_group_uids)
        if not selected:
            return []
        fields = batch.non_tensor_batch
        group_values = np.asarray(fields.get("uid", []), dtype=object)
        row_indices = []
        seen = set()
        for i, group_uid_raw in enumerate(group_values):
            group_uid = str(group_uid_raw)
            if group_uid not in selected or group_uid in seen:
                continue
            row_indices.append(i)
            seen.add(group_uid)
            if len(seen) == len(selected):
                break
        return row_indices

    def _failure_reflection_cfg(self):
        cfg = self._teacher_rollout_cfg()
        if cfg is None:
            return {}
        return cfg.get("failure_reflection", {}) or {}

    def _failure_reflection_enabled(self) -> bool:
        return bool(self._failure_reflection_cfg().get("enabled", False))

    def _collect_failed_student_traces(self, student_batch, selected_group_uids, cfg):
        selected = {str(g) for g in selected_group_uids}
        if not selected:
            return {}
        fields = student_batch.non_tensor_batch
        group_values = np.asarray(fields.get("uid", []), dtype=object)
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        max_failed_trajs = max(int(cfg.get("max_failed_trajs", 2)), 0)
        if max_failed_trajs <= 0:
            return {}
        max_steps = max(int(cfg.get("max_steps_per_traj", 4)), 1)
        threshold = float((self._teacher_rollout_cfg() or {}).get("success_reward_threshold", 1.0))

        grouped = {}
        order = []
        for i, group_uid_raw in enumerate(group_values):
            group_uid = str(group_uid_raw)
            if group_uid not in selected:
                continue
            traj_uid = str(traj_values[i]) if len(traj_values) else f"row-{i}"
            key = (group_uid, traj_uid)
            reward = self._float_value(self._field_at(fields, "episode_rewards", i, 0.0))
            if reward >= threshold:
                continue
            if key not in grouped:
                grouped[key] = {"group_uid": group_uid, "traj_uid": traj_uid, "reward": reward, "steps": []}
                order.append(key)
            grouped[key]["reward"] = reward
            if len(grouped[key]["steps"]) >= max_steps:
                continue
            grouped[key]["steps"].append({
                "step_idx": self._int_value(self._field_at(fields, "step_idx", i), len(grouped[key]["steps"]) + 1),
                "action": self._field_at(fields, "model_response", i, self._field_at(fields, "response", i)),
                "observation": self._field_at(fields, "observation", i),
                "next_observation": self._field_at(fields, "next_observation", i),
                "valid": self._field_at(fields, "is_action_valid", i, True),
            })

        traces_by_group = {}
        for key in order:
            trace = grouped[key]
            group_uid = trace["group_uid"]
            traces = traces_by_group.setdefault(group_uid, [])
            if len(traces) >= max_failed_trajs:
                continue
            trace["steps"].sort(key=lambda step: step["step_idx"])
            traces.append(trace)
        return traces_by_group

    def _format_failure_reflection(self, group_uid: str, traces, cfg) -> str:
        max_chars = max(int(cfg.get("max_chars", 1600)), 0)
        if max_chars <= 0 or not traces:
            return ""
        include_observations = bool(cfg.get("include_observations", False))
        include_invalid_actions = bool(cfg.get("include_invalid_actions", True))
        lines = [
            "## Previous Failed Student Attempts",
            "The no-skill student attempted this task and received low reward. Use these traces only to avoid repeating failed searches or unsupported final answers.",
            "",
        ]
        for attempt_idx, trace in enumerate(traces, start=1):
            lines.append(f"Attempt {attempt_idx} (traj_uid={trace['traj_uid']}):")
            lines.append(f"- Previous final reward: {trace['reward']:.3g}")
            for step in trace["steps"]:
                step_idx = step["step_idx"]
                action = self._short_text(step.get("action"), 360)
                if action:
                    lines.append(f"- Step {step_idx} action: {action}")
                valid = step.get("valid", True)
                if include_invalid_actions and str(valid).lower() in {"false", "0"}:
                    lines.append(f"- Step {step_idx} invalid action: true")
                if include_observations:
                    obs = self._short_text(step.get("observation"), 360)
                    next_obs = self._short_text(step.get("next_observation"), 360)
                    if obs:
                        lines.append(f"- Step {step_idx} observation: {obs}")
                    if next_obs:
                        lines.append(f"- Step {step_idx} next observation: {next_obs}")
            lines.append("- Note: the previous final answer/action was marked incorrect.")
            lines.append("")
        lines.append("Guidance: try a different search query or verify missing entities/relations before answering.")
        reflection = "\n".join(lines).strip()
        return reflection if len(reflection) <= max_chars else reflection[: max(max_chars - 3, 0)].rstrip() + "..."

    def _build_failure_reflections(self, student_batch, selected_group_uids):
        cfg = self._failure_reflection_cfg()
        if not bool(cfg.get("enabled", False)):
            return {}
        traces_by_group = self._collect_failed_student_traces(student_batch, selected_group_uids, cfg)
        return {
            group_uid: reflection
            for group_uid, traces in traces_by_group.items()
            if (reflection := self._format_failure_reflection(group_uid, traces, cfg))
        }

    def _build_teacher_env_kwargs(self, student_batch, selected_rows, teacher_n: int):
        fields = student_batch.non_tensor_batch
        if "extra.gamefile" in fields:
            env_kwargs = []
            for row in selected_rows:
                gamefile = fields["extra.gamefile"][row]
                if gamefile is None:
                    return None
                env_kwargs.extend({"extra.gamefile": gamefile} for _ in range(teacher_n))
            return env_kwargs
        for replay_key in ("session", "task_id"):
            if replay_key in fields:
                env_kwargs = []
                for row in selected_rows:
                    replay_id = fields[replay_key][row]
                    if replay_id is None:
                        return None
                    env_kwargs.extend({"session": replay_id, "_skill_int_explicit_sessions": True} for _ in range(teacher_n))
                return env_kwargs
        if "env_kwargs" in fields:
            env_kwargs = []
            for row in selected_rows:
                value = fields["env_kwargs"][row]
                env_kwargs.extend(deepcopy(value) for _ in range(teacher_n))
            return env_kwargs
        return None

    def _build_teacher_gen_batch(self, original_gen_batch, student_batch, selected_group_uids):
        cfg = self._teacher_rollout_cfg()
        if cfg is None:
            return None
        teacher_n = max(self._curriculum_teacher_n(cfg), 1)
        student_rollout_n = max(int(self.config.env.rollout.n), 1)
        selected_rows = self._select_first_traj_rows_by_group(student_batch, selected_group_uids)
        if not selected_rows:
            return None
        try:
            prompt_indices = [int(student_batch.non_tensor_batch["index"][row]) // student_rollout_n for row in selected_rows]
        except Exception as err:
            print(f"[SkillSDTrainer] teacher rollout skipped: cannot map student rows to prompts: {err}", flush=True)
            return None

        env_kwargs = self._build_teacher_env_kwargs(student_batch, selected_rows, teacher_n)
        if env_kwargs is None:
            print("[SkillSDTrainer] teacher rollout skipped: no replayable task ids.", flush=True)
            return None

        failure_reflections = self._build_failure_reflections(student_batch, selected_group_uids)
        teacher_prompt_indices = []
        teacher_group_uids = []
        teacher_reflections = []
        for prompt_idx, row in zip(prompt_indices, selected_rows):
            group_uid = str(student_batch.non_tensor_batch["uid"][row])
            reflection = failure_reflections.get(group_uid)
            teacher_prompt_indices.extend([prompt_idx] * teacher_n)
            teacher_group_uids.extend([group_uid] * teacher_n)
            teacher_reflections.extend([reflection] * teacher_n)
        real_teacher_count = len(teacher_prompt_indices)

        try:
            target_env_num = _infer_env_num(getattr(self, "envs", None))
        except Exception:
            target_env_num = real_teacher_count
        if real_teacher_count > 0 and target_env_num > real_teacher_count:
            pad_count = target_env_num - real_teacher_count
            teacher_prompt_indices.extend([teacher_prompt_indices[-1]] * pad_count)
            teacher_group_uids.extend([teacher_group_uids[-1]] * pad_count)
            teacher_reflections.extend([teacher_reflections[-1]] * pad_count)
            env_kwargs.extend([deepcopy(env_kwargs[-1]) for _ in range(pad_count)])

        for i, reflection in enumerate(teacher_reflections):
            if not reflection:
                continue
            if not isinstance(env_kwargs[i], dict):
                env_kwargs[i] = {"_base_env_kwargs": env_kwargs[i]}
            env_kwargs[i]["skill_int_failure_reflection"] = reflection

        teacher_gen_batch = original_gen_batch.select_idxs(np.asarray(teacher_prompt_indices, dtype=np.int64))
        teacher_gen_batch.non_tensor_batch["env_kwargs"] = np.asarray(env_kwargs, dtype=object)
        teacher_gen_batch.non_tensor_batch["skill_int_failure_reflection"] = np.asarray(teacher_reflections, dtype=object)
        teacher_gen_batch.meta_info["skill_int_teacher_group_uids"] = teacher_group_uids
        teacher_gen_batch.meta_info["skill_int_teacher_real_count"] = real_teacher_count
        teacher_gen_batch.meta_info["skill_int_teacher_source_rows"] = list(selected_rows)
        teacher_gen_batch.meta_info["skill_int_failure_reflection_count"] = int(sum(1 for value in failure_reflections.values() if value))
        teacher_gen_batch.meta_info["skill_int_failure_reflection_chars_mean"] = (
            float(np.mean([len(value) for value in failure_reflections.values() if value]))
            if failure_reflections else 0.0
        )
        return teacher_gen_batch

    def _annotate_teacher_batch(self, teacher_batch, teacher_gen_batch):
        group_uids = list(teacher_gen_batch.meta_info.get("skill_int_teacher_group_uids", []))
        if not group_uids:
            return teacher_batch
        indexes = np.asarray(teacher_batch.non_tensor_batch.get("index", []), dtype=int)
        if indexes.size:
            mapped = [group_uids[int(i)] if int(i) < len(group_uids) else str(group_uids[-1]) for i in indexes]
        else:
            mapped = list(group_uids)
        teacher_batch.non_tensor_batch["uid"] = np.asarray(mapped, dtype=object)
        teacher_batch.non_tensor_batch["arm"] = np.asarray(["teacher"] * len(mapped), dtype=object)

        gen_reflections = np.asarray(teacher_gen_batch.non_tensor_batch.get("skill_int_failure_reflection", []), dtype=object)
        if gen_reflections.size:
            reflections = [gen_reflections[int(i)] if int(i) < len(gen_reflections) else None for i in indexes] if indexes.size else list(gen_reflections[: len(mapped)])
            teacher_batch.non_tensor_batch["skill_int_failure_reflection"] = np.asarray(reflections, dtype=object)
        teacher_batch.meta_info["skill_int_teacher_group_uids"] = group_uids
        return teacher_batch

    def _drop_teacher_padding_rows(self, teacher_batch, teacher_gen_batch):
        real_count = int(teacher_gen_batch.meta_info.get("skill_int_teacher_real_count", len(teacher_gen_batch.batch)))
        indexes = np.asarray(teacher_batch.non_tensor_batch.get("index", []), dtype=int)
        if indexes.size == 0 or real_count >= len(teacher_gen_batch.batch):
            return teacher_batch
        keep_rows = np.nonzero(indexes < real_count)[0]
        if keep_rows.size == len(indexes):
            return teacher_batch
        return teacher_batch.select_idxs(keep_rows.astype(np.int64))

    def _active_state_query_group_utilities(self, decisions) -> Dict[str, float]:
        cfg = self._teacher_rollout_cfg() or {}
        aux_gate_mode = str(cfg.get("aux_gate_mode", "success"))
        utility_eps = float(cfg.get("utility_eps", 0.0))
        use_group_utility = bool(cfg.get("aux_use_group_utility", True))
        return {
            str(group_uid): float(decision.utility) if use_group_utility else 0.0
            for group_uid, decision in decisions.items()
            if _teacher_aux_gate_active(
                aux_gate_mode,
                bool(decision.teacher_success),
                float(decision.utility),
                utility_eps,
            )
        }

    def _state_query_group_qualities(self, decisions) -> Dict[str, float]:
        success_threshold = max(
            float((self._teacher_rollout_cfg() or {}).get("success_reward_threshold", 10.0)),
            1e-8,
        )
        qualities = {}
        for group_uid, decision in decisions.items():
            if isinstance(decision, dict):
                task_score = decision.get("teacher_task_score_best", None)
                reward = decision.get("teacher_reward_best", None)
            else:
                task_score = getattr(decision, "teacher_task_score_best", None)
                reward = getattr(decision, "teacher_reward_best", None)
            if task_score is None and reward is None:
                quality = 1.0
            else:
                quality = max(float(task_score or 0.0), float(reward or 0.0) / success_threshold)
            qualities[str(group_uid)] = float(np.clip(quality, 0.0, 1.0))
        return qualities

    def _build_state_teacher_query_prompt_batch(self, student_batch, records):
        if not records:
            return None
        rows = torch.as_tensor([int(record["student_row"]) for record in records], dtype=torch.long)
        alt_input_ids = student_batch.batch["alt_input_ids"][rows]
        alt_attention_mask = student_batch.batch["alt_attention_mask"][rows]
        alt_position_ids = student_batch.batch.get("alt_position_ids")
        selected_position_ids = (
            alt_position_ids[rows] if alt_position_ids is not None
            else compute_position_id_with_mask(alt_attention_mask)
        )
        tensors = {
            "input_ids": alt_input_ids,
            "attention_mask": alt_attention_mask,
            "position_ids": selected_position_ids,
        }
        meta_info = dict(student_batch.meta_info)
        meta_info["do_sample"] = bool(self._state_teacher_query_cfg().get("do_sample", False))
        meta_info["validate"] = False
        meta_info["cpsd_teacher_lora_requested"] = True
        return DataProto.from_dict(tensors=tensors, meta_info=meta_info)

    def _attach_state_query_source_context(self, query_batch, student_batch, records):
        rows = torch.as_tensor([int(record["student_row"]) for record in records], dtype=torch.long)
        response_len = int(student_batch.batch["responses"].shape[-1])
        if "prompts" in student_batch.batch.keys():
            no_skill_input_ids = student_batch.batch["prompts"][rows]
            prompt_len = int(no_skill_input_ids.shape[-1])
        else:
            prompt_len = int(student_batch.batch["input_ids"].shape[-1]) - response_len
            no_skill_input_ids = student_batch.batch["input_ids"][rows, :prompt_len]
        no_skill_attention_mask = student_batch.batch["attention_mask"][rows, :prompt_len]
        no_skill_position_ids = student_batch.batch["position_ids"][rows, ..., :prompt_len]
        query_batch.batch["alt_input_ids"] = no_skill_input_ids
        query_batch.batch["alt_attention_mask"] = no_skill_attention_mask
        query_batch.batch["alt_position_ids"] = no_skill_position_ids

        source_rows = np.asarray([int(record["student_row"]) for record in records], dtype=np.int64)
        source_fields = student_batch.non_tensor_batch
        non_tensors = {
            key: np.asarray(values[source_rows], dtype=object)
            for key, values in source_fields.items()
        }
        responses = self.tokenizer.batch_decode(query_batch.batch["responses"], skip_special_tokens=True)
        source_trajs = [str(record["student_traj_uid"]) for record in records]
        source_steps = [int(record["student_step"]) for record in records]
        source_no_skill_prompts = [self._field_at(source_fields, "prompt_text", row) for row in source_rows]
        source_skill_prompts = [
            record.get("teacher_query_prompt_text")
            or self._field_at(source_fields, "alt_prompt_text", row)
            for record, row in zip(records, source_rows)
        ]
        non_tensors["uid"] = np.asarray([record["group_uid"] for record in records], dtype=object)
        non_tensors["traj_uid"] = np.asarray(
            [f"{traj}:state_teacher_query:{step}:{idx}" for idx, (traj, step) in enumerate(zip(source_trajs, source_steps))],
            dtype=object,
        )
        non_tensors["arm"] = np.asarray(["state_teacher_query"] * len(records), dtype=object)
        non_tensors["model_response"] = np.asarray(responses, dtype=object)
        non_tensors["prompt_text"] = np.asarray(source_skill_prompts, dtype=object)
        non_tensors["alt_prompt_text"] = np.asarray(source_no_skill_prompts, dtype=object)
        non_tensors["rewards"] = np.asarray([0.0] * len(records), dtype=object)
        non_tensors["episode_rewards"] = np.asarray([0.0] * len(records), dtype=object)
        non_tensors["is_action_valid"] = np.asarray([None] * len(records), dtype=object)
        non_tensors["skill_int_state_query_source_row"] = source_rows.astype(object)
        non_tensors["skill_int_state_query_source_traj_uid"] = np.asarray(source_trajs, dtype=object)
        non_tensors["skill_int_state_query_source_step"] = np.asarray(source_steps, dtype=object)
        non_tensors["skill_int_state_query_source_reward"] = np.asarray(
            [record["student_episode_reward"] for record in records], dtype=object
        )
        non_tensors["skill_int_state_query_source_action_valid"] = np.asarray(
            [record["student_action_valid"] for record in records], dtype=object
        )
        non_tensors["skill_int_state_query_selection_reason"] = np.asarray(
            [record["selection_reason"] for record in records], dtype=object
        )
        non_tensors["skill_int_state_query_utility"] = np.asarray(
            [record["utility"] for record in records], dtype=object
        )
        non_tensors["skill_int_state_query_quality"] = np.asarray(
            [record["quality"] for record in records], dtype=object
        )
        query_batch.non_tensor_batch = non_tensors
        query_batch.meta_info = dict(student_batch.meta_info)
        query_batch.meta_info["skill_int_state_query_records"] = records
        query_batch.meta_info["skill_int_state_query_utilities"] = [float(record["utility"]) for record in records]
        query_batch.meta_info["skill_int_state_query_qualities"] = [float(record["quality"]) for record in records]
        return query_batch

    def _dump_state_teacher_query_batch(self, query_batch, records) -> None:
        if query_batch is None or not records or not bool(self._state_teacher_query_cfg().get("dump", True)):
            return
        path = self._default_skill_sd_dir("state_teacher_queries")
        if path is None:
            return
        path = os.path.join(path, f"step_{int(getattr(self, 'global_steps', 0)):06d}_queries.jsonl")
        responses = self.tokenizer.batch_decode(query_batch.batch["responses"], skip_special_tokens=True)
        for idx, (record, response) in enumerate(zip(records, responses)):
            self._append_jsonl(path, {
                "global_step": int(getattr(self, "global_steps", 0)),
                **record,
                "teacher_response": response,
                "teacher_action": cf_action_at(query_batch, idx),
                "student_prompt": self._field_at(query_batch.non_tensor_batch, "alt_prompt_text", idx),
                "teacher_prompt": self._field_at(query_batch.non_tensor_batch, "prompt_text", idx),
            })

    def _run_student_state_teacher_queries(self, student_batch, decisions):
        if not self._state_teacher_query_enabled():
            return {"skill_int/state_query_enabled": 0.0}, None
        group_utilities = self._active_state_query_group_utilities(decisions)
        group_qualities = self._state_query_group_qualities(decisions)
        records = self._select_student_state_query_records(student_batch, group_utilities, group_qualities)
        metrics = {
            "skill_int/state_query_enabled": 1.0,
            "skill_int/state_query_active_groups": float(len(group_utilities)),
            "skill_int/state_query_selected_groups": float(len({r["group_uid"] for r in records})),
            "skill_int/state_query_selected_states": float(len(records)),
        }
        if not records:
            if any(key not in student_batch.batch.keys() for key in ("alt_input_ids", "alt_attention_mask")):
                metrics["skill_int/state_query_missing_alt_prompt"] = 1.0
            return metrics, None

        prompt_batch = self._build_state_teacher_query_prompt_batch(student_batch, records)
        padded, pad_size = pad_dataproto_to_divisor(prompt_batch, max(int(self.actor_rollout_wg.world_size), 1))
        try:
            output = self.actor_rollout_wg.generate_sequences(padded)
            output = unpad_dataproto(output, pad_size=pad_size)
            query_batch = self._attach_state_query_source_context(output, student_batch, records)
        except Exception as err:
            print(f"[SkillSDTrainer] state teacher query failed (non-fatal): {err}", flush=True)
            metrics["skill_int/state_query_failed"] = 1.0
            return metrics, None

        response_len = int(query_batch.batch["responses"].shape[-1])
        response_mask = query_batch.batch["attention_mask"][:, -response_len:]
        decoded = self.tokenizer.batch_decode(query_batch.batch["responses"], skip_special_tokens=True)
        keep = np.asarray([
            bool(text.strip()) and bool(response_mask[idx].sum().item())
            for idx, text in enumerate(decoded)
        ])
        if not keep.all():
            keep_rows = np.nonzero(keep)[0].astype(np.int64)
            query_batch = query_batch.select_idxs(keep_rows)
            records = [record for record, keep_row in zip(records, keep.tolist()) if keep_row]
            query_batch.meta_info["skill_int_state_query_records"] = records
            query_batch.meta_info["skill_int_state_query_utilities"] = [float(record["utility"]) for record in records]
            query_batch.meta_info["skill_int_state_query_qualities"] = [float(record["quality"]) for record in records]

        metrics["skill_int/state_query_generated_rows"] = float(len(query_batch.batch))
        metrics["skill_int/state_query_empty_rows"] = float(len(keep) - int(keep.sum()))
        metrics["skill_int/state_query_step_mean"] = float(np.mean([r["student_step"] for r in records])) if records else 0.0
        metrics["skill_int/state_query_invalid_source_frac"] = (
            float(np.mean([not r["student_action_valid"] for r in records])) if records else 0.0
        )
        if not records:
            return metrics, None
        self._dump_state_teacher_query_batch(query_batch, records)
        print(
            "[SkillSDTrainer] generated one-step teacher labels on student states: "
            f"groups={int(metrics['skill_int/state_query_selected_groups'])}, rows={len(query_batch.batch)}",
            flush=True,
        )
        return metrics, query_batch

    def _run_teacher_intervention_rollout(self, original_loop, original_gen_batch, student_batch, selected_group_uids, student_stats):
        if getattr(self, "_skill_int_running_teacher", False):
            return {}, None
        teacher_gen_batch = self._build_teacher_gen_batch(original_gen_batch, student_batch, selected_group_uids)
        if teacher_gen_batch is None:
            return {}, None
        print(
            "[SkillSDTrainer] running teacher intervention rollout: "
            f"groups={len(selected_group_uids)}, rows={len(teacher_gen_batch.batch)}",
            flush=True,
        )
        self._skill_int_running_teacher = True
        try:
            with _force_teacher_alt_prompt_tensors(self.envs, self.traj_collector, env_num=len(teacher_gen_batch.batch)):
                teacher_batch = original_loop(
                    gen_batch=teacher_gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.envs,
                    is_train=False,
                    progress_label=f"teacher intervention step {getattr(self, 'global_steps', 0)}",
                )
        finally:
            self._skill_int_running_teacher = False
        teacher_batch = self._drop_teacher_padding_rows(teacher_batch, teacher_gen_batch)
        teacher_batch = self._annotate_teacher_batch(teacher_batch, teacher_gen_batch)

        cfg = self._teacher_rollout_cfg()
        success_threshold = float(cfg.get("success_reward_threshold", 10.0))
        utility_mode = str(cfg.get("utility_mode", "best_minus_student_mean"))
        task_score_utility = bool(cfg.get("task_score_utility", False))
        teacher_stats = compute_teacher_group_stats(teacher_batch, success_reward_threshold=success_threshold)
        decisions = compute_teacher_intervention_decisions(
            student_stats,
            teacher_stats,
            selected_group_uids=selected_group_uids,
            utility_mode=utility_mode,
            task_score_utility=task_score_utility,
        )
        metrics = summarize_teacher_intervention(decisions)
        metrics["skill_int/failure_reflection_enabled"] = float(self._failure_reflection_enabled())
        metrics["skill_int/failure_reflection_groups"] = float(teacher_gen_batch.meta_info.get("skill_int_failure_reflection_count", 0))
        metrics["skill_int/failure_reflection_chars_mean"] = float(teacher_gen_batch.meta_info.get("skill_int_failure_reflection_chars_mean", 0.0))
        if decisions:
            used = [d for d in decisions.values() if d.teacher_used]
            if used:
                metrics["skill_int/teacher_reward_best_mean"] = float(np.mean([d.teacher_reward_best for d in used]))
                metrics["skill_int/teacher_student_mean_gap"] = float(np.mean([d.teacher_reward_best - d.student_reward_mean for d in used]))
                metrics["skill_int/teacher_task_score_best_mean"] = float(np.mean([d.teacher_task_score_best for d in used]))
                metrics["skill_int/teacher_student_task_gap"] = float(np.mean([d.teacher_task_score_best - d.student_task_score_mean for d in used]))
            aux_gate_mode = str(cfg.get("aux_gate_mode", "success"))
            utility_eps = float(cfg.get("utility_eps", 0.0))
            active = [
                d for d in decisions.values()
                if _teacher_aux_gate_active(aux_gate_mode, bool(d.teacher_success), float(d.utility), utility_eps)
            ]
            metrics["skill_int/active_teacher_groups"] = float(len(active))
            metrics["skill_int/active_teacher_frac"] = float(len(active) / max(len(decisions), 1))
            metrics["skill_int/teacher_n_current"] = float(self._curriculum_teacher_n(cfg))

        teacher_batch.meta_info["skill_int_teacher_decisions"] = {
            group_uid: decision.__dict__ for group_uid, decision in decisions.items()
        }
        state_query_metrics, state_query_batch = self._run_student_state_teacher_queries(student_batch, decisions)
        metrics.update(state_query_metrics)
        self._last_skill_int_state_query_batch = state_query_batch
        cf_pairs = self._collect_and_annotate_cf_span_pairs(student_batch, teacher_batch, decisions)
        successful_teacher_pool = {
            (str(group_uid), str(traj_uid))
            for group_uid, decision in decisions.items()
            for traj_uid in decision.successful_teacher_traj_uids
        }
        cf_teacher_pool = {(pair.group_uid, pair.teacher_traj_uid) for pair in cf_pairs}
        used_successful_teacher_pool = cf_teacher_pool & successful_teacher_pool
        metrics["skill_int/cf_successful_teacher_trajs_used"] = float(len(used_successful_teacher_pool))
        metrics["skill_int/cf_successful_teacher_traj_use_frac"] = float(
            len(used_successful_teacher_pool) / max(len(successful_teacher_pool), 1)
        )
        metrics["skill_int/cf_max_teacher_pairs_per_student"] = float(
            self.skill_sd_cfg.get("cf_max_teacher_pairs_per_student", 3) or 0
        )
        if cf_pairs:
            pair_groups = {pair.group_uid for pair in cf_pairs}
            div_turns = np.asarray([pair.divergence_turn for pair in cf_pairs], dtype=float)
            div_steps = np.asarray([pair.teacher_step for pair in cf_pairs], dtype=float)
            reward_gaps = np.asarray([pair.utility for pair in cf_pairs], dtype=float)
            div_depths = np.asarray([
                (max(float(pair.teacher_step), 1.0) - float(pair.divergence_turn)) / max(float(pair.teacher_step), 1.0)
                for pair in cf_pairs
            ], dtype=float)
            metrics["skill_int/cf_candidate_pairs"] = float(len(cf_pairs))
            metrics["skill_int/cf_candidate_groups"] = float(len(pair_groups))
            metrics["skill_int/cf_reward_gap_mean"] = float(reward_gaps.mean())
            metrics["skill_int/cf_reward_gap_min"] = float(reward_gaps.min())
            metrics["skill_int/first_diff_turn_idx_mean"] = float(div_turns.mean())
            metrics["skill_int/first_diff_turn_idx_min"] = float(div_turns.min())
            metrics["skill_int/first_diff_turn_idx_max"] = float(div_turns.max())
            metrics["skill_int/first_diff_turn_step_mean"] = float(div_steps.mean())
            metrics["skill_int/first_diff_depth_mean"] = float(div_depths.mean())
            # divergence-anchored suffix diagnostics (one pair/group).
            div_ratios = np.asarray([getattr(p, "critical_divergence_ratio", 0.0) for p in cf_pairs], dtype=float)
            suffix_lens = np.asarray([getattr(p, "successful_suffix_length", 0) for p in cf_pairs], dtype=float)
            metrics["skill_int/critical_divergence_step"] = float(div_turns.mean())
            metrics["skill_int/critical_divergence_ratio"] = float(div_ratios.mean())
            metrics["skill_int/successful_suffix_length"] = float(suffix_lens.mean())

        # rescue rate = selected all-fail groups with >=1 successful
        # teacher traj (decisions only holds selected all-fail groups under hard_only).
        rescued_groups = [d for d in decisions.values() if getattr(d, "successful_teacher_traj_uids", ())]
        n_selected = max(len(decisions), 1)
        metrics["skill_int/skill_rescue_attempts"] = float(len(decisions))
        metrics["skill_int/skill_rescue_rate"] = float(len(rescued_groups) / n_selected)

        self._dump_teacher_trajectory_batch(teacher_batch, decisions)
        self._dump_cf_span_pairs(student_batch, teacher_batch, cf_pairs)
        self._log_task_success_trajectories(student_batch, arm="student")
        self._log_task_success_trajectories(teacher_batch, arm="teacher", decisions=decisions)
        self._log_intervention_dynamic_csv(student_stats, decisions, metrics)
        return metrics, teacher_batch

    # ------------------------------------------------------------------ #
    # Teacher / CF dumps
    # ------------------------------------------------------------------ #

    def _resolve_teacher_dump_dir(self) -> Optional[str]:
        cfg = self._teacher_rollout_cfg()
        if cfg is not None and cfg.get("dump_dir", None):
            return str(cfg.get("dump_dir"))
        return self._default_skill_sd_dir("teacher_trajectories")

    def _dump_teacher_trajectory_batch(self, teacher_batch, decisions) -> None:
        cfg = self._teacher_rollout_cfg()
        if cfg is None or not bool(self.skill_sd_cfg.get("dump_teacher_trajectories", True)):
            return
        dump_dir = self._resolve_teacher_dump_dir()
        if dump_dir is None:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"step_{self.global_steps:06d}_teacher_trajectories.jsonl")
            responses = self.tokenizer.batch_decode(teacher_batch.batch["responses"], skip_special_tokens=True)
            fields = teacher_batch.non_tensor_batch
            records: Dict[str, Dict[str, Any]] = {}
            order = []
            for row_idx, response in enumerate(responses):
                traj_uid = str(self._json_safe(self._field_at(fields, "traj_uid", row_idx, f"row-{row_idx}")))
                if traj_uid not in records:
                    group_uid = str(self._json_safe(self._field_at(fields, "uid", row_idx)))
                    decision = decisions.get(group_uid)
                    records[traj_uid] = {
                        "global_step": int(getattr(self, "global_steps", 0)),
                        "uid": group_uid,
                        "traj_uid": traj_uid,
                        "arm": "teacher",
                        "task_id": self._json_safe(self._task_identity(fields, row_idx)),
                        "data_source": self._json_safe(self._field_at(fields, "data_source", row_idx)),
                        "episode_reward": self._json_safe(self._field_at(fields, "episode_rewards", row_idx)),
                        "episode_task_score": self._json_safe(self._field_at(fields, "episode_task_scores", row_idx, 0.0)),
                        "episode_length": self._json_safe(self._field_at(fields, "episode_lengths", row_idx)),
                        "tool_callings": self._json_safe(self._field_at(fields, "tool_callings", row_idx)),
                        "failure_reflection": self._json_safe(self._field_at(fields, "skill_int_failure_reflection", row_idx)),
                        "decision": self._json_safe(decision.__dict__ if decision is not None else None),
                        "steps": [],
                    }
                    order.append(traj_uid)
                records[traj_uid]["steps"].append({
                    "step": self._int_value(self._field_at(fields, "step_idx", row_idx), len(records[traj_uid]["steps"]) + 1),
                    "prompt": self._json_safe(self._field_at(fields, "prompt_text", row_idx)),
                    "observation": self._json_safe(self._field_at(fields, "observation", row_idx)),
                    "response": response,
                    "model_response": self._json_safe(self._field_at(fields, "model_response", row_idx, response)),
                    "reward": self._json_safe(self._field_at(fields, "rewards", row_idx)),
                    "done": self._json_safe(self._field_at(fields, "dones", row_idx)),
                    "lora_requested": self._json_safe(
                        self._field_at(fields, "lora_requested", row_idx, False)
                    ),
                    "active": self._json_safe(self._field_at(fields, "active_masks", row_idx)),
                    "is_action_valid": self._json_safe(self._field_at(fields, "is_action_valid", row_idx)),
                    "next_observation": self._json_safe(self._field_at(fields, "next_observation", row_idx)),
                })
            with open(path, "a", encoding="utf-8") as fh:
                for traj_uid in order:
                    records[traj_uid]["steps"].sort(key=lambda item: item.get("step", 0))
                    fh.write(json.dumps(records[traj_uid], ensure_ascii=False) + "\n")
            print(f"[SkillSDTrainer] dumped {len(order)} teacher trajectories -> {path}", flush=True)
        except Exception as err:
            print(f"[SkillSDTrainer] teacher trajectory dump failed (non-fatal): {err}", flush=True)

    def _should_collect_cf_span_pairs(self) -> bool:
        return bool(self.skill_sd_cfg.get("enable_cf", False) or self.skill_sd_cfg.get("dump_span_pairs", True))

    def _collect_and_annotate_cf_span_pairs(self, student_batch, teacher_batch, decisions):
        if not self._should_collect_cf_span_pairs():
            teacher_batch.meta_info["skill_int_cf_span_pairs"] = []
            return []
        teacher_cfg = self._teacher_rollout_cfg() or {}
        pairs = collect_student_first_cf_span_pairs(
            student_batch=student_batch,
            teacher_batch=teacher_batch,
            decisions=decisions,
            divergence_mode=str(self.skill_sd_cfg.get("cf_divergence_mode", "first")),
            aux_gate_mode=str(teacher_cfg.get("aux_gate_mode", "success")),
            utility_eps=float(teacher_cfg.get("utility_eps", 0.0)),
            max_teacher_pairs_per_student=self.skill_sd_cfg.get("cf_max_teacher_pairs_per_student", 3),
            pair_mode=str(self.skill_sd_cfg.get("cf_pair_mode", "teacher_better")),
            utility_source=str(self.skill_sd_cfg.get("cf_utility_source", "reward")),
            utility_scale=float(self.skill_sd_cfg.get("cf_utility_scale", 1.0)),
            rescue_pairing=bool(self.skill_sd_cfg.get("rescue_pairing", False)),
        )
        self._annotate_teacher_cf_span_pairs(teacher_batch, pairs)
        return pairs

    def _annotate_teacher_cf_span_pairs(self, teacher_batch, pairs):
        n_rows = len(teacher_batch.batch)
        pair_ids = np.asarray([None] * n_rows, dtype=object)
        student_rows = np.asarray([-1] * n_rows, dtype=object)
        student_trajs = np.asarray([None] * n_rows, dtype=object)
        divergence_turns = np.asarray([-1] * n_rows, dtype=object)
        utilities = np.asarray([0.0] * n_rows, dtype=object)
        teacher_task_ids = np.asarray([None] * n_rows, dtype=object)
        student_task_ids = np.asarray([None] * n_rows, dtype=object)
        suffix_rows_mask = np.zeros(n_rows, dtype=bool)  # L_suffix demo rows
        for pair in pairs:
            row = int(pair.teacher_row_idx)
            if row < 0 or row >= n_rows:
                continue
            pair_ids[row] = pair.pair_id
            student_rows[row] = int(pair.student_row_idx)
            student_trajs[row] = pair.student_traj_uid
            divergence_turns[row] = int(pair.divergence_turn)
            utilities[row] = float(pair.utility)
            teacher_task_ids[row] = pair.teacher_task_id
            student_task_ids[row] = pair.student_task_id
            for suffix_row in getattr(pair, "suffix_teacher_rows", ()) or ():
                sr = int(suffix_row)
                if 0 <= sr < n_rows:
                    suffix_rows_mask[sr] = True
        teacher_batch.non_tensor_batch["skill_int_cf_pair_id"] = pair_ids
        teacher_batch.non_tensor_batch["skill_int_cf_student_row"] = student_rows
        teacher_batch.non_tensor_batch["skill_int_cf_student_traj_uid"] = student_trajs
        teacher_batch.non_tensor_batch["skill_int_cf_divergence_turn"] = divergence_turns
        teacher_batch.non_tensor_batch["skill_int_cf_utility"] = utilities
        teacher_batch.non_tensor_batch["skill_int_cf_teacher_task_id"] = teacher_task_ids
        teacher_batch.non_tensor_batch["skill_int_cf_student_task_id"] = student_task_ids
        teacher_batch.non_tensor_batch["skill_int_rescue_suffix_row"] = suffix_rows_mask.astype(object)
        teacher_batch.meta_info["skill_int_cf_span_pairs"] = [pair.__dict__ for pair in pairs]
        return teacher_batch

    def _resolve_span_dump_dir(self) -> Optional[str]:
        cfg = self._teacher_rollout_cfg()
        if cfg is not None and cfg.get("span_dump_dir", None):
            return str(cfg.get("span_dump_dir"))
        return self._default_skill_sd_dir("span_pairs")

    def _dump_cf_span_pairs(self, student_batch, teacher_batch, pairs) -> None:
        if not pairs or not bool(self.skill_sd_cfg.get("dump_span_pairs", True)):
            return
        dump_dir = self._resolve_span_dump_dir()
        if dump_dir is None:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"step_{self.global_steps:06d}_cf_span_pairs.jsonl")
            teacher_responses = self.tokenizer.batch_decode(teacher_batch.batch["responses"], skip_special_tokens=True)
            student_responses = self.tokenizer.batch_decode(student_batch.batch["responses"], skip_special_tokens=True)
            t_fields = teacher_batch.non_tensor_batch
            s_fields = student_batch.non_tensor_batch
            with open(path, "a", encoding="utf-8") as fh:
                for pair in pairs:
                    t_row = int(pair.teacher_row_idx)
                    s_row = int(pair.student_row_idx)
                    record = {
                        "global_step": int(getattr(self, "global_steps", 0)),
                        "pair_id": pair.pair_id,
                        "uid": pair.group_uid,
                        "utility": float(pair.utility),
                        "group_utility": self._json_safe(pair.group_utility),
                        "divergence_turn": int(pair.divergence_turn),
                        "teacher": {
                            "traj_uid": pair.teacher_traj_uid,
                            "row_idx": t_row,
                            "step": int(pair.teacher_step),
                            "task_id": self._json_safe(self._task_identity(t_fields, t_row)),
                            "prompt": self._json_safe(self._field_at(t_fields, "prompt_text", t_row)),
                            "observation": self._json_safe(self._field_at(t_fields, "observation", t_row)),
                            "action": pair.teacher_action,
                            "response": teacher_responses[t_row] if 0 <= t_row < len(teacher_responses) else None,
                            "response_hash": pair.teacher_response_hash,
                            "reward": self._json_safe(self._field_at(t_fields, "rewards", t_row)),
                            "episode_reward": self._json_safe(self._field_at(t_fields, "episode_rewards", t_row)),
                        },
                        "student": {
                            "traj_uid": pair.student_traj_uid,
                            "row_idx": s_row,
                            "step": int(pair.student_step),
                            "task_id": self._json_safe(self._task_identity(s_fields, s_row)),
                            "prompt": self._json_safe(self._field_at(s_fields, "prompt_text", s_row)),
                            "observation": self._json_safe(self._field_at(s_fields, "observation", s_row)),
                            "action": pair.student_action,
                            "response": student_responses[s_row] if 0 <= s_row < len(student_responses) else None,
                            "response_hash": pair.student_response_hash,
                            "reward": self._json_safe(self._field_at(s_fields, "rewards", s_row)),
                            "episode_reward": self._json_safe(self._field_at(s_fields, "episode_rewards", s_row)),
                        },
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[SkillSDTrainer] dumped {len(pairs)} CF span pairs -> {path}", flush=True)
        except Exception as err:
            print(f"[SkillSDTrainer] CF span pair dump failed (non-fatal): {err}", flush=True)

    # ------------------------------------------------------------------ #
    # Train rollout and aux-row attachment
    # ------------------------------------------------------------------ #

    def _install_student_first_rollout_hook(self) -> None:
        train_envs = getattr(self, "envs", None)
        traj_collector = getattr(self, "traj_collector", None)
        if train_envs is None or traj_collector is None:
            print("[SkillSDTrainer] teacher_rollout enabled but envs/traj_collector missing; skipping rollout hook.")
            return
        original_loop = traj_collector.multi_turn_loop
        student_rollout_cfg = self.skill_sd_cfg.get("student_rollout", {}) or {}
        student_use_skill_static = bool(student_rollout_cfg.get("use_skill", False))
        skillrl_warmup_steps = max(int(self.skill_sd_cfg.get("skillrl_warmup_steps", 0)), 0)

        def _student_use_skill_now() -> bool:
            if student_use_skill_static:
                return True
            return skillrl_warmup_steps > 0 and int(getattr(self, "global_steps", 0)) < skillrl_warmup_steps

        def student_first_multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train=True, **kw):
            if is_train and envs is train_envs:
                self._last_skill_int_teacher_batch = None
                self._last_skill_int_state_query_batch = None
                self._last_skill_int_cf_branch_proto = None
                self._last_skill_int_prefix_rescue_proto = None
                self._last_skill_int_prefix_rescue_suffix_proto = None
            student_use_skill = _student_use_skill_now()
            if not is_train or envs is not train_envs or student_use_skill:
                ctx = nullcontext()
            elif (
                self._state_teacher_query_enabled()
                or self._cf_branch_enabled()
                or self._prefix_rescue_enabled()
            ) and getattr(train_envs, "retrieval_memory", None) is not None:
                ctx = _force_student_alt_prompt_tensors(train_envs, traj_collector, env_num=len(gen_batch.batch))
            else:
                ctx = _disable_retrieval_memory(train_envs)
            with ctx:
                out = original_loop(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs, is_train=is_train, **kw)
            if is_train and envs is train_envs:
                self._archive_trajectory_batch(
                    out, record_type="train_student_trajectory", arm="student"
                )
                metrics, selected_group_uids, student_stats = self._compute_student_intervention_stats(out)
                prefix_group_uids = self._prefix_rescue_candidate_group_uids(
                    student_stats, selected_group_uids
                )
                metrics = dict(metrics) if metrics else {}
                metrics["skill_int/student_use_skill_active"] = float(student_use_skill)
                warmup_active = (not student_use_skill_static) and student_use_skill
                metrics["skill_int/skillrl_warmup_active"] = float(warmup_active)
                prev = out.meta_info.get("skill_int_metrics", {})
                prev.update(metrics)
                out.meta_info["skill_int_metrics"] = prev
                out.meta_info["skill_int_selected_group_uids"] = list(selected_group_uids)
                if metrics:
                    print_keys = [
                        "skill_int/student_reward_mean",
                        "skill_int/student_success_frac",
                        "skill_int/teacher_candidate_groups",
                        "skill_int/teacher_candidate_frac",
                    ]
                    print(
                        "[SkillSDTrainer] student-first intervention candidates: "
                        + ", ".join(f"{k}={metrics[k]:.3f}" for k in print_keys if k in metrics),
                        flush=True,
                    )
                if warmup_active:
                    out.meta_info["skill_int_metrics"]["skill_int/teacher_skipped_in_warmup"] = 1.0
                elif prefix_group_uids and self._prefix_rescue_enabled():
                    prefix_metrics, prefix_proto = self._run_prefix_rescue_rollout(
                        out, prefix_group_uids
                    )
                    out.meta_info.setdefault("skill_int_metrics", {}).update(prefix_metrics)
                    if prefix_proto is not None:
                        self._last_skill_int_prefix_rescue_proto = prefix_proto
                replace_teacher = bool(self._prefix_rescue_cfg().get("replace_teacher_rollout", True))
                if (
                    not warmup_active
                    and selected_group_uids
                    and self._teacher_rollout_enabled()
                    and not (self._prefix_rescue_enabled() and replace_teacher)
                ):
                    teacher_metrics, teacher_batch = self._run_teacher_intervention_rollout(
                        original_loop=original_loop,
                        original_gen_batch=gen_batch,
                        student_batch=out,
                        selected_group_uids=selected_group_uids,
                        student_stats=student_stats,
                    )
                    if teacher_metrics:
                        out.meta_info["skill_int_metrics"].update(teacher_metrics)
                    if teacher_batch is not None:
                        self._last_skill_int_teacher_batch = teacher_batch
                        out.meta_info["skill_int_teacher_available"] = True
                if self._cf_branch_enabled():
                    # Must run BEFORE the alt-prompt pop below (needs alt_input_ids).
                    cf_metrics, cf_pair_proto = self._run_cf_branch_rollout(out)
                    out.meta_info.setdefault("skill_int_metrics", {}).update(cf_metrics)
                    if cf_pair_proto is not None:
                        self._last_skill_int_cf_branch_proto = cf_pair_proto
                if (
                    self._state_teacher_query_enabled()
                    or self._cf_branch_enabled()
                    or self._prefix_rescue_enabled()
                ) and not bool(self.skill_sd_cfg.get("legacy_alt_prompt_tensors", False)):
                    for key in ("alt_input_ids", "alt_attention_mask", "alt_position_ids"):
                        if key in out.batch.keys():
                            out.batch.pop(key)
                    out.non_tensor_batch.pop("alt_prompt_text", None)
                if self._process_enabled():
                    phi_metrics = self._apply_phi_reward_shaping(out)
                    out.meta_info.setdefault("skill_int_metrics", {}).update(phi_metrics)
            return out

        traj_collector.multi_turn_loop = student_first_multi_turn_loop
        self._student_first_rollout_hook_installed = True
        print(
            "[SkillSDTrainer] student-first teacher intervention hook installed "
            f"(student_use_skill_static={student_use_skill_static}, skillrl_warmup_steps={skillrl_warmup_steps})."
        )

    def _process_cfg(self):
        if self.skill_sd_cfg is None:
            return {}
        return self.skill_sd_cfg.get("process", {}) or {}

    def _process_enabled(self) -> bool:
        return bool(self._process_cfg().get("enabled", False))

    def _phi_env_name(self) -> Optional[str]:
        env_name = str(getattr(self.config, "env", {}).get("env_name", "")).lower()
        if env_name == "webshop":
            return "webshop"
        if env_name.startswith("alfworld"):
            return "alfworld"
        return None

    def _prefix_env_name(self) -> Optional[str]:
        env = self._phi_env_name()
        if env is not None:
            return env
        env_name = str(getattr(self.config, "env", {}).get("env_name", "")).lower()
        return "searchqa" if "search" in env_name else None

    def _apply_phi_reward_shaping(self, out) -> Dict[str, float]:
        """Apply optional potential shaping to per-step environment rewards."""
        env = self._phi_env_name()
        if env is None:
            return {"cpsd/phi_shaping_skipped_env": 1.0}
        ntb = out.non_tensor_batch
        if "rewards" not in ntb or "traj_uid" not in ntb:
            return {"cpsd/phi_shaping_skipped_missing_fields": 1.0}

        n = len(ntb["traj_uid"])
        traj_uids = [str(x) for x in ntb["traj_uid"]]
        step_idxs = [int(x) for x in ntb.get("step_idx", list(range(n)))]
        observations = list(ntb.get("observation", [""] * n))
        actions = list(ntb.get("model_response", [""] * n))
        rewards = [float(x) for x in ntb["rewards"]]

        # task descriptor per trajectory (WebShop instruction / ALFWorld gamefile).
        task_of_traj: Dict[str, str] = {}
        terminal_reward_of_traj: Dict[str, float] = {}
        episode_rewards = ntb.get("episode_rewards", None)
        for i in range(n):
            tuid = traj_uids[i]
            if tuid not in task_of_traj:
                if env == "webshop":
                    task_of_traj[tuid] = str(
                        self._field_at(ntb, "instruction_text", i, "")
                        or self._field_at(ntb, "session", i, "")
                    )
                else:  # alfworld: milestone regex needs the gamefile PATH, not task_id
                    task_of_traj[tuid] = self._alfworld_gamefile(ntb, i)
            if episode_rewards is not None and tuid not in terminal_reward_of_traj:
                try:
                    terminal_reward_of_traj[tuid] = float(episode_rewards[i])
                except (TypeError, ValueError):
                    pass

        new_rewards, metrics = shape_rewards_with_phi(
            env=env,
            traj_uids=traj_uids,
            step_idxs=step_idxs,
            observations=observations,
            actions=actions,
            rewards=rewards,
            task_of_traj=task_of_traj,
            terminal_reward_of_traj=terminal_reward_of_traj,
            eta=float(self._process_cfg().get("eta", 1.0)),
            gamma=float(self._process_cfg().get("gamma", 0.95)),
        )
        # Store the per-step reward alongside each environment transition.
        out.non_tensor_batch["rewards"] = np.asarray(new_rewards, dtype=object)
        metrics["cpsd/phi_shaping_active"] = 1.0
        return metrics

    def _dump_prefix_rescue_student_groups(self, student_batch, clusters: list[SharedPrefixCluster]) -> None:
        cfg = self._prefix_rescue_cfg()
        if not clusters or not bool(cfg.get("dump_student_groups", True)):
            return
        path = self._default_skill_sd_dir("prefix_rescue/student_groups")
        if path is None:
            return
        path = os.path.join(path, f"step_{int(getattr(self, 'global_steps', 0)):06d}_student_groups.jsonl")
        fields = student_batch.non_tensor_batch
        group_values = np.asarray(fields.get("uid", []), dtype=object)
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        responses = self.tokenizer.batch_decode(student_batch.batch["responses"], skip_special_tokens=True)
        clusters_by_group: Dict[str, list[tuple[int, SharedPrefixCluster]]] = {}
        for scheduled_order, cluster in enumerate(clusters):
            clusters_by_group.setdefault(str(cluster.group_uid), []).append(
                (scheduled_order, cluster)
            )
        selected_groups = set(clusters_by_group)
        grouped: Dict[str, Dict[str, list[int]]] = {}
        for row, group_uid_raw in enumerate(group_values):
            group_uid = str(group_uid_raw)
            if group_uid not in selected_groups:
                continue
            traj_uid = str(traj_values[row])
            grouped.setdefault(group_uid, {}).setdefault(traj_uid, []).append(row)
        for group_uid, trajectories in grouped.items():
            scheduled_clusters = clusters_by_group[group_uid]
            frontier_clusters = []
            for group_scheduled_order, (scheduled_order, cluster) in enumerate(
                scheduled_clusters
            ):
                cluster_payload = cluster.to_dict()
                cluster_payload["scheduled_order"] = int(scheduled_order)
                cluster_payload["group_scheduled_order"] = int(
                    group_scheduled_order
                )
                frontier_clusters.append(cluster_payload)
            payload = {
                "global_step": int(getattr(self, "global_steps", 0)),
                "group_uid": group_uid,
                # Keep the first scheduled frontier under the legacy key while
                # preserving the complete ordered frontier below.
                "cluster": frontier_clusters[0],
                "frontier_cluster_count": len(frontier_clusters),
                "frontier_clusters": frontier_clusters,
                "scheduled_frontier_node_uids": [
                    str(cluster.state_hash)
                    for _scheduled_order, cluster in scheduled_clusters
                ],
                "trajectories": [],
            }
            for traj_uid, rows in sorted(trajectories.items()):
                rows.sort(key=lambda row: self._int_value(self._field_at(fields, "step_idx", row), row + 1))
                matched_frontiers = [
                    (scheduled_order, cluster)
                    for scheduled_order, cluster in scheduled_clusters
                    if traj_uid in cluster.matched_traj_uids
                ]
                payload["trajectories"].append({
                    "traj_uid": traj_uid,
                    "episode_reward": self._float_value(self._field_at(fields, "episode_rewards", rows[0], 0.0)),
                    "matched": bool(matched_frontiers),
                    "matched_frontier_node_uids": [
                        str(cluster.state_hash)
                        for _scheduled_order, cluster in matched_frontiers
                    ],
                    "matched_frontier_scheduled_orders": [
                        int(scheduled_order)
                        for scheduled_order, _cluster in matched_frontiers
                    ],
                    "steps": [
                        {
                            "step": self._int_value(self._field_at(fields, "step_idx", row), 0),
                            "observation": self._field_at(fields, "observation", row),
                            "model_response": self._field_at(fields, "model_response", row, responses[row]),
                            "response": responses[row],
                            "next_observation": self._field_at(fields, "next_observation", row),
                            "reward": self._field_at(fields, "rewards", row),
                            "done": self._field_at(fields, "dones", row),
                        }
                        for row in rows
                    ],
                })
            self._append_jsonl(path, payload)

    def _dump_prefix_rescue_diagnostics(
        self, clusters, attempts_by_group, selected_by_group, teacher_context_by_group=None
    ) -> None:
        if not clusters or not bool(self._prefix_rescue_cfg().get("dump", True)):
            return
        path = self._default_skill_sd_dir("prefix_rescue")
        if path is None:
            return
        path = os.path.join(path, f"step_{int(getattr(self, 'global_steps', 0)):06d}_diagnostics.jsonl")
        teacher_context_by_group = teacher_context_by_group or {}
        for cluster in clusters:
            selected = selected_by_group.get(cluster.group_uid)
            self._append_jsonl(path, {
                "global_step": int(getattr(self, "global_steps", 0)),
                "cluster": cluster.to_dict(),
                "teacher_context": teacher_context_by_group.get(cluster.group_uid, {}),
                "attempts": attempts_by_group.get(cluster.group_uid, []),
                "selected_attempt": selected,
            })

    def _prefix_rescue_session(self, fields, row: int):
        env = self._prefix_env_name()
        if env == "alfworld":
            value = self._field_at(fields, "extra.gamefile", row, None) \
                or self._field_at(fields, "task_id", row, None)
            return None if value is None else str(value)
        if env == "searchqa":
            value = self._field_at(fields, "env_kwargs", row, None)
            if not isinstance(value, dict) or not value.get("question") or "ground_truth" not in value:
                return None
            return deepcopy(value)
        value = self._field_at(fields, "session", row, None) \
            or self._field_at(fields, "task_id", row, None)
        if value is None:
            return None
        try:
            return int(str(value).split(":")[-1])
        except (TypeError, ValueError):
            return None

    def _build_prefix_rescue_suffix_proto(self, student_batch, examples):
        """Encode the same verified suffix under skill and no-skill contexts."""
        if not examples:
            return None
        response_len = int(student_batch.batch["responses"].shape[-1])
        prompt_len = int(student_batch.batch["input_ids"].shape[-1]) - response_len
        if prompt_len <= 0:
            return None
        pad_id = int(self.tokenizer.pad_token_id)
        def pack(key, max_len, left_pad):
            ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, example in enumerate(examples):
                tokens = torch.as_tensor(example[key], dtype=torch.long)
                if not 0 < len(tokens) <= max_len:
                    raise ValueError(f"Invalid {key} length {len(tokens)} for width {max_len}")
                start = max_len - len(tokens) if left_pad else 0
                ids[row, start:start + len(tokens)] = tokens
                mask[row, start:start + len(tokens)] = 1
            return ids, mask

        prompt_ids, prompt_mask = pack("student_prompt_ids", prompt_len, True)
        skill_prompt_ids, skill_prompt_mask = pack("teacher_prompt_ids", prompt_len, True)
        response_ids, response_mask = pack("response_ids", response_len, False)
        response_texts = [example["response"] for example in examples]
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        skill_attention_mask = torch.cat([skill_prompt_mask, response_mask], dim=-1)
        tensors = {
            "input_ids": torch.cat([prompt_ids, response_ids], dim=-1),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "responses": response_ids,
            "response_mask": response_mask.clone(),
            "skill_int_suffix_q_input_ids": torch.cat(
                [skill_prompt_ids, response_ids], dim=-1
            ),
            "skill_int_suffix_q_attention_mask": skill_attention_mask,
            "skill_int_suffix_q_position_ids": compute_position_id_with_mask(
                skill_attention_mask
            ),
            "skill_int_suffix_action_mask": response_mask.clone(),
            "skill_int_suffix_archive_node_hash": torch.as_tensor([
                int(
                    hashlib.sha1(
                        str(example.get("node_uid", "")).encode("utf-8")
                    ).hexdigest()[:15],
                    16,
                )
                for example in examples
            ], dtype=torch.long),
            "skill_int_suffix_archive_step": torch.as_tensor([
                self._int_value(example.get("environment_step", -1), -1)
                for example in examples
            ], dtype=torch.long),
        }
        non_tensors = {
            "uid": np.asarray([str(example["group_uid"]) for example in examples], dtype=object),
            "arm": np.asarray(["prefix_rescue_suffix"] * len(examples), dtype=object),
            "model_response": np.asarray(
                response_texts,
                dtype=object,
            ),
        }
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=dict(student_batch.meta_info),
        )

    def _prefix_rescue_first_divergence_groups(
        self,
        student_batch,
        job: FailurePrefixJob,
        result: dict,
        first_teacher_proto,
        *,
        consumed_traj_uids,
    ):
        """Match each student to its first aligned divergence on a success path."""
        fields = student_batch.non_tensor_batch
        group_values = np.asarray(fields.get("uid", []), dtype=object)
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        step_values = np.asarray(fields.get("step_idx", []), dtype=object)
        rows_by_traj: Dict[str, list[int]] = {}
        member_set = {str(uid) for uid in job.matched_traj_uids}
        for row in range(len(student_batch.batch)):
            if str(group_values[row]) != str(job.group_uid):
                continue
            traj_uid = str(traj_values[row])
            if traj_uid in member_set:
                rows_by_traj.setdefault(traj_uid, []).append(row)
        for rows in rows_by_traj.values():
            rows.sort(
                key=lambda row: self._int_value(
                    step_values[row] if row < len(step_values) else row + 1,
                    row + 1,
                )
            )

        teacher_actions = list(result.get("actions", []) or [])
        teacher_observations = list(result.get("observations", []) or [])
        suffix_examples = list(result.get("no_skill_suffix_examples", []) or [])
        consumed = {str(uid) for uid in consumed_traj_uids}
        grouped: Dict[int, list[tuple[str, int, float]]] = {}
        same_action_rows = 0
        state_mismatch_rows = 0
        reward_by_traj = dict(
            zip(job.matched_traj_uids, job.matched_episode_rewards)
        )
        for traj_uid in job.matched_traj_uids:
            traj_uid = str(traj_uid)
            if traj_uid in consumed:
                continue
            rows = rows_by_traj.get(traj_uid, [])
            max_turn = min(len(teacher_actions), len(rows))
            for turn in range(int(job.depth), max_turn):
                student_row = int(rows[turn])
                teacher_state = " ".join(
                    str(
                        teacher_observations[turn]
                        if turn < len(teacher_observations) else ""
                    ).strip().lower().split()
                )
                student_state = " ".join(
                    str(cf_observation_at(student_batch, student_row) or "")
                    .strip().lower().split()
                )
                if not teacher_state or teacher_state != student_state:
                    state_mismatch_rows += 1
                    break
                student_action = cf_action_at(student_batch, student_row)
                teacher_action = str(teacher_actions[turn] or "")
                if not student_action or not teacher_action:
                    break
                if not _cf_branch.actions_diverge(teacher_action, student_action):
                    same_action_rows += 1
                    continue
                if (not self._field_at(fields, "is_action_valid", student_row, False)
                        or not self._prefix_rescue_action_mask(student_batch.select_idxs([student_row])).any()):
                    break
                grouped.setdefault(turn, []).append((
                    traj_uid,
                    student_row,
                    float(reward_by_traj[traj_uid]),
                ))
                break

        divergence_groups = []
        for turn, members in sorted(grouped.items()):
            if turn == int(job.depth):
                teacher_proto = first_teacher_proto
            else:
                suffix_idx = turn - int(job.depth) - 1
                if suffix_idx < 0 or suffix_idx >= len(suffix_examples):
                    continue
                suffix_example = {
                    **dict(suffix_examples[suffix_idx]),
                    "group_uid": f"{job.group_uid}:{job.node_uid}:t{turn}",
                }
                teacher_proto = self._build_prefix_rescue_suffix_proto(
                    student_batch, [suffix_example]
                )
                if teacher_proto is None:
                    continue
            if teacher_proto is None or not self._prefix_rescue_action_mask(teacher_proto).any():
                continue
            local_node_uid = f"{job.node_uid}:t{turn}"
            member_uids = tuple(item[0] for item in members)
            member_rows = tuple(item[1] for item in members)
            member_rewards = tuple(item[2] for item in members)
            divergence_groups.append({
                "job": replace(
                    job,
                    node_uid=local_node_uid,
                    depth=int(turn),
                    support=len(members),
                    representative_traj_uid=member_uids[0],
                    representative_row=member_rows[0],
                    matched_traj_uids=member_uids,
                    matched_rows=member_rows,
                    matched_episode_rewards=member_rewards,
                ),
                "teacher_proto": teacher_proto,
                "teacher_action": str(teacher_actions[turn] or ""),
                "turn": int(turn),
                # suffix_examples[k] is teacher turn depth+1+k.  Each local
                # divergence owns its own strictly-later conditional KL rows.
                "suffix_examples": list(suffix_examples[int(turn) - int(job.depth):]),
            })
        return (
            divergence_groups,
            same_action_rows,
            state_mismatch_rows,
        )

    @staticmethod
    def _failure_job_overlaps_active(job, active_jobs) -> bool:
        members = {str(uid) for uid in job.matched_traj_uids}
        return any(
            str(active.group_uid) == str(job.group_uid)
            and bool(members.intersection(
                str(uid) for uid in active.matched_traj_uids
            ))
            for active in active_jobs
        )

    def _run_prefix_rescue_jobs(self, student_batch, tries, frontier_jobs, metrics):
        """Try continuations at each frontier node, then back off to an eligible ancestor."""
        cfg = self._prefix_rescue_cfg()
        fields = student_batch.non_tensor_batch
        min_support = max(int(cfg.get("min_support", 2)), 2)
        max_attempts = max(int(cfg.get("attempts", 4)), 1)
        root_retry_n = max(int(cfg.get("root_retry_n", 1)), 1)
        success_threshold = float(
            (self._teacher_rollout_cfg() or {}).get("success_reward_threshold", 10.0)
        )
        trie_by_group = {trie.group_uid: trie for trie in tries}
        consumed = {trie.group_uid: set() for trie in tries}
        attempted_nodes = {trie.group_uid: set() for trie in tries}
        attempts_used = {trie.group_uid: 0 for trie in tries}
        nonroot_attempts_used = {trie.group_uid: 0 for trie in tries}
        root_attempts_used = {trie.group_uid: 0 for trie in tries}
        attempts_by_group: Dict[str, list[dict]] = {
            trie.group_uid: [] for trie in tries
        }
        successes = []
        node_attempts = {}
        pending = list(frontier_jobs)
        valid_teacher_actions = 0
        teacher_suffix_generations = 0
        teacher_suffix_valid_actions = 0
        teacher_suffix_invalid_actions = 0
        teacher_suffix_retries = 0
        teacher_suffix_aborted_attempts = 0
        replay_verified = 0
        same_action_rows = 0
        state_mismatch_rows = 0
        self._cf_branch_gen_meta = dict(student_batch.meta_info)
        frontier_ids = {
            (job.group_uid, job.node_uid) for job in frontier_jobs
        }
        for trie in tries:
            for node in trie.nodes:
                self._append_artifact("failure_prefix_node", {
                    **node.to_dict(),
                    "candidate_kind": str(trie.candidate_kind),
                    "is_frontier": (node.group_uid, node.node_uid) in frontier_ids,
                })

        while pending:
            active = []
            active_attempt_indices = {}
            deferred = []
            wave_deferred = []
            round_nodes = set()
            active_groups = set()
            for raw_job in sorted(
                pending,
                key=lambda job: (
                    job.group_uid,
                    -int(job.depth),
                    -int(job.support),
                    job.node_uid,
                ),
            ):
                group_uid = str(raw_job.group_uid)
                if group_uid in active_groups:
                    wave_deferred.append(raw_job)
                    continue
                filtered = filter_failure_prefix_job_members(
                    raw_job, consumed[group_uid], min_support=min_support
                )
                key = (group_uid, raw_job.node_uid)
                if filtered is None or key in round_nodes:
                    continue
                if raw_job.node_uid in attempted_nodes[group_uid]:
                    continue
                if self._failure_job_overlaps_active(filtered, active):
                    wave_deferred.append(filtered)
                    continue
                round_nodes.add(key)
                attempt_index = attempts_used[group_uid]
                attempts_used[group_uid] += 1
                phase_attempt = node_attempts.get(key, 0) + 1
                node_attempts[key] = phase_attempt
                if filtered.is_root:
                    root_attempts_used[group_uid] += 1
                    phase, phase_budget = "root", root_retry_n
                else:
                    nonroot_attempts_used[group_uid] += 1
                    phase, phase_budget = "nonroot", max_attempts
                active.append(filtered)
                active_groups.add(group_uid)
                active_attempt_indices[key] = attempt_index
                self._append_artifact("teacher_rescue_job", {
                    **filtered.to_dict(),
                    "attempt": attempt_index,
                    "attempt_phase": phase,
                    "candidate_kind": str(
                        trie_by_group[group_uid].candidate_kind
                    ),
                    "phase_attempt": phase_attempt,
                    "phase_budget": phase_budget,
                    "nonroot_attempt_budget": max_attempts,
                    "root_attempt_budget": root_retry_n,
                    "consumed_before": sorted(consumed[group_uid]),
                })
            if not active:
                dedup = {}
                for job in wave_deferred + deferred:
                    dedup.setdefault((job.group_uid, job.node_uid), job)
                pending = list(dedup.values())
                if pending:
                    continue
                break

            specs, spec_meta = [], []
            failed_jobs, replay_failures = [], set()
            for job in active:
                attempt_index = active_attempt_indices[
                    (str(job.group_uid), job.node_uid)
                ]
                attempt_record = {
                    "attempt": attempt_index,
                    "attempt_phase": "root" if bool(job.is_root) else "nonroot",
                    "candidate_kind": str(
                        trie_by_group[job.group_uid].candidate_kind
                    ),
                    "phase_attempt": node_attempts[(job.group_uid, job.node_uid)],
                    "phase_budget": root_retry_n if job.is_root else max_attempts,
                    "node_uid": job.node_uid,
                    "parent_uid": job.parent_uid,
                    "is_frontier": (
                        (job.group_uid, job.node_uid) in frontier_ids
                    ),
                    "prefix_depth": int(job.depth),
                    "prefix_support": int(job.support),
                    "teacher_action": "",
                    "replay_verified": False,
                    "done": False,
                    "terminal_reward": 0.0,
                    "success": False,
                    "signal_emitted": False,
                    "status": "scheduled",
                    "branch_actions": [],
                    "branch_observations": [],
                }
                attempts_by_group[job.group_uid].append(attempt_record)
                self._print_rescue_job(job, attempt_record, matched_students=0)
                task_desc = str(
                    self._field_at(
                        fields, "instruction_text", int(job.representative_row), ""
                    )
                    or ""
                )
                if not task_desc and self._prefix_env_name() == "alfworld":
                    task_desc = self._alfworld_task_description(
                        fields,
                        int(job.representative_row),
                        job.expected_observations[0]
                        if job.expected_observations else "",
                    )
                skill_text = _cf_branch.extract_skill_text(
                    self._field_at(
                        fields, "alt_prompt_text", int(job.representative_row), ""
                    )
                )
                attempt_record["task_description"] = task_desc
                attempt_record["skill_text"] = skill_text
                attempt_record["skill_text_hash"] = hashlib.sha1(
                    skill_text.encode("utf-8")
                ).hexdigest()[:16]
                session = self._prefix_rescue_session(fields, int(job.representative_row))
                if session is None:
                    replay_failures.add((job.group_uid, job.node_uid))
                    attempt_record["status"] = "missing_session"
                    self._print_rescue_job(job, attempt_record, matched_students=0)
                    self._archive_prefix_rescue_attempt(
                        student_batch, job, attempt_record
                    )
                    failed_jobs.append(job)
                    continue
                attempt_record["status"] = "rollout_started"
                if not task_desc and isinstance(session, dict):
                    task_desc = str(session.get("question", "") or "")
                specs.append({
                    "session": session,
                    "task_id": str(job.task_id),
                    "script": list(job.prefix_actions),
                    "launch_prompts": (
                        self._field_at(fields, "prompt_text", job.representative_row, ""),
                        self._field_at(fields, "alt_prompt_text", job.representative_row, ""),
                    ),
                    "prefix_len": int(job.depth),
                    "task_desc": task_desc,
                    "role": "teacher",
                    "skill_text": skill_text,
                    "expected_observations": job.expected_observations,
                    "track_phi": False,
                })
                spec_meta.append((job, attempt_record))
            branch_results = []
            if specs:
                self._skill_int_running_prefix_rescue = True
                try:
                    branch_results = self._run_branches_breadth_first(
                        specs,
                        n_cont=int(getattr(self.config.env, "max_steps", 50)),
                        max_steps=int(getattr(self.config.env, "max_steps", 50)),
                    )
                finally:
                    self._skill_int_running_prefix_rescue = False

            for result, (job, attempt_record) in zip(branch_results, spec_meta, strict=True):
                result = result or {}
                generations = result.get("no_skill_suffix_examples", [])
                launch = generations[0] if generations else {}
                teacher_action = launch.get("action", "")
                valid_teacher_actions += int(bool(launch.get("action_valid")))
                attempt_record.update({
                    "teacher_action": teacher_action,
                    "teacher_query_prompt": launch.get("skill_prompt"),
                    "teacher_query_no_skill_prompt": launch.get("no_skill_prompt"),
                    "teacher_query_raw_response": launch.get("response"),
                })
                result["no_skill_suffix_examples"] = generations[1:]
                teacher_suffix_generations += int(
                    result.get("suffix_generation_count", 0) if result else 0
                )
                teacher_suffix_valid_actions += int(
                    result.get("suffix_valid_action_count", 0) if result else 0
                )
                teacher_suffix_invalid_actions += int(
                    result.get("suffix_invalid_generation_count", 0)
                    if result else 0
                )
                teacher_suffix_retries += int(
                    result.get("suffix_retry_count", 0) if result else 0
                )
                generation_failed = bool(
                    result and result.get("generation_failed", False)
                )
                teacher_suffix_aborted_attempts += int(generation_failed)
                verified = bool(result and result.get("replay_verified", False))
                replay_verified += int(verified)
                terminal_reward = float(
                    result.get("terminal_reward", 0.0) if result else 0.0
                )
                terminal_success = bool(
                    verified and result.get("done", False) and terminal_reward >= success_threshold
                )
                attempt_record["outcome_success"] = terminal_success
                attempt_record.update({
                    "replay_verified": verified,
                    "done": bool(result.get("done", False)) if result else False,
                    "terminal_reward": terminal_reward,
                    "terminal_task_score": float(
                        result.get("terminal_task_score", 0.0) if result else 0.0
                    ),
                    "terminal_success": terminal_success,
                    "teacher_suffix_generation_failed": generation_failed,
                    "teacher_suffix_generation_count": int(
                        result.get("suffix_generation_count", 0) if result else 0
                    ),
                    "teacher_suffix_valid_action_count": int(
                        result.get("suffix_valid_action_count", 0) if result else 0
                    ),
                    "teacher_suffix_invalid_generation_count": int(
                        result.get("suffix_invalid_generation_count", 0)
                        if result else 0
                    ),
                    "teacher_suffix_retry_count": int(
                        result.get("suffix_retry_count", 0) if result else 0
                    ),
                    "success": False,
                    "status": "terminal_success_pending_match" if terminal_success else (
                        "replay_mismatch" if not verified else (
                            "invalid_teacher_suffix"
                            if generation_failed else "rollout_failed"
                        )
                    ),
                    "branch_actions": list(result.get("actions", []) if result else []),
                    "branch_observations": list(
                        result.get("observations", []) if result else []
                    ),
                    "steps": list(result.get("steps", []) if result else []),
                })
                query_turn = int(job.depth)
                if query_turn < len(attempt_record["steps"]):
                    attempt_record["steps"][query_turn].update({
                        "skill_prompt": attempt_record.get("teacher_query_prompt"),
                        "no_skill_prompt": attempt_record.get(
                            "teacher_query_no_skill_prompt"
                        ),
                        "raw_response": attempt_record.get(
                            "teacher_query_raw_response"
                        ),
                        "parsed_action": teacher_action,
                        "skill_text_hash": attempt_record.get("skill_text_hash"),
                    })
                if not terminal_success:
                    self._print_rescue_job(job, attempt_record, matched_students=0)
                    self._archive_prefix_rescue_attempt(
                        student_batch, job, attempt_record
                    )
                    if not verified:
                        replay_failures.add((job.group_uid, job.node_uid))
                    failed_jobs.append(job)
                    continue

                first_teacher_proto = self._build_prefix_rescue_suffix_proto(
                    student_batch, [{**launch, "group_uid": job.group_uid}]
                )
                (
                    divergence_groups,
                    same_count,
                    mismatch_count,
                ) = self._prefix_rescue_first_divergence_groups(
                    student_batch,
                    job,
                    result or {},
                    first_teacher_proto,
                    consumed_traj_uids=consumed[job.group_uid],
                )
                same_action_rows += int(same_count)
                state_mismatch_rows += int(mismatch_count)
                if not divergence_groups:
                    attempt_record["status"] = "success_without_divergence"
                    attempt_record["matched_students"] = 0
                    self._print_rescue_job(job, attempt_record, matched_students=0)
                    self._archive_prefix_rescue_attempt(
                        student_batch, job, attempt_record
                    )
                    attempted_nodes[job.group_uid].add(job.node_uid)
                    continue
                matched_students = sum(
                    group["job"].support for group in divergence_groups
                )
                for group in divergence_groups:
                    consumed[job.group_uid].update(
                        group["job"].matched_traj_uids
                    )
                attempt_record["signal_emitted"] = True
                attempt_record["success"] = True
                attempt_record["status"] = "success"
                attempt_record["matched_students"] = int(matched_students)
                attempted_nodes[job.group_uid].add(job.node_uid)
                for group in divergence_groups:
                    signal_job = group["job"]
                    local_group_uid = (
                        f"{job.group_uid}:{signal_job.node_uid}"
                    )
                    successes.append({
                        "group_uid": job.group_uid,
                        "local_group_uid": local_group_uid,
                        "job": signal_job,
                        "attempt": int(attempt_record["attempt"]),
                        "teacher_action": group["teacher_action"],
                        "terminal_reward": terminal_reward,
                        "teacher_proto": group["teacher_proto"],
                        "suffix_examples": group["suffix_examples"],
                    })
                self._print_rescue_job(
                    job, attempt_record, matched_students=matched_students
                )
                self._archive_prefix_rescue_attempt(
                    student_batch, job, attempt_record
                )

            # A failed leaf backs off to the nearest untried eligible ancestor.
            # Converged ancestors are deduplicated before the next batch.
            for job in failed_jobs:
                group_uid = job.group_uid
                key = (group_uid, job.node_uid)
                limit = root_retry_n if job.is_root else max_attempts
                if key not in replay_failures and node_attempts[key] < limit:
                    deferred.append(job)
                    continue
                attempted_nodes[group_uid].add(job.node_uid)
                parent = next_failure_prefix_backoff_job(
                    trie_by_group[group_uid], job.node_uid,
                    consumed_traj_uids=consumed[group_uid],
                    attempted_node_uids=attempted_nodes[group_uid],
                    min_support=min_support,
                )
                if parent is not None:
                    deferred.append(parent)
            dedup = {}
            for job in wave_deferred + deferred:
                dedup.setdefault((job.group_uid, job.node_uid), job)
            pending = list(dedup.values())

        metrics["prefix_rescue/teacher_attempts"] = float(sum(attempts_used.values()))
        metrics["prefix_rescue/teacher_nonroot_attempts"] = float(
            sum(nonroot_attempts_used.values())
        )
        metrics["prefix_rescue/teacher_root_attempts"] = float(
            sum(root_attempts_used.values())
        )
        metrics["prefix_rescue/nonroot_attempt_budget"] = float(max_attempts)
        metrics["prefix_rescue/root_attempt_budget"] = float(
            root_retry_n if root_retry_n is not None else 1
        )
        metrics["prefix_rescue/max_attempts_per_nonroot_node"] = float(max_attempts)
        metrics["prefix_rescue/max_attempts_at_root"] = float(root_retry_n)
        metrics["prefix_rescue/valid_teacher_actions"] = float(valid_teacher_actions)
        metrics["prefix_rescue/teacher_suffix_generations"] = float(
            teacher_suffix_generations
        )
        metrics["prefix_rescue/teacher_suffix_valid_actions"] = float(
            teacher_suffix_valid_actions
        )
        metrics["prefix_rescue/teacher_suffix_invalid_actions"] = float(
            teacher_suffix_invalid_actions
        )
        metrics["prefix_rescue/teacher_suffix_valid_action_rate"] = float(
            teacher_suffix_valid_actions / max(teacher_suffix_generations, 1)
        )
        metrics["prefix_rescue/teacher_suffix_retries"] = float(
            teacher_suffix_retries
        )
        metrics["prefix_rescue/teacher_suffix_aborted_attempts"] = float(
            teacher_suffix_aborted_attempts
        )
        metrics["prefix_rescue/replay_verified_attempts"] = float(replay_verified)
        metrics["prefix_rescue/teacher_success_jobs"] = float(
            sum(
                int(record.get("success", False))
                for records in attempts_by_group.values()
                for record in records
            )
        )
        metrics["prefix_rescue/pairs_emitted"] = float(
            sum(success["job"].support for success in successes)
        )
        metrics["prefix_rescue/consumed_student_trajectories"] = float(
            sum(len(items) for items in consumed.values())
        )
        metrics["prefix_rescue/same_action_rows"] = float(same_action_rows)
        metrics["prefix_rescue/state_mismatch_rows"] = float(state_mismatch_rows)
        initial_depth_by_group = {}
        for job in frontier_jobs:
            initial_depth_by_group[job.group_uid] = max(
                int(job.depth),
                initial_depth_by_group.get(job.group_uid, 0),
            )
        backoff_depths_by_group = {}
        root_groups = set()
        success_groups = set()
        success_root_groups = set()
        success_natural_groups = set()
        backoff_success_groups = set()
        first_success_attempts = []
        for group_uid, records in attempts_by_group.items():
            if not records:
                continue
            depths = [int(record.get("prefix_depth", 0)) for record in records]
            backoff_depths = [
                int(record.get("prefix_depth", 0))
                for record in records
                if not bool(record.get("is_frontier", False))
            ]
            if backoff_depths:
                backoff_depths_by_group[group_uid] = backoff_depths
            if 0 in depths:
                root_groups.add(group_uid)
            successful = [
                record for record in records if bool(record.get("success", False))
            ]
            if successful:
                success_groups.add(group_uid)
                if any(
                    int(record.get("prefix_depth", 0)) == 0
                    for record in successful
                ):
                    success_root_groups.add(group_uid)
                if any(
                    int(record.get("prefix_depth", 0)) > 0
                    for record in successful
                ):
                    success_natural_groups.add(group_uid)
                first_success_attempts.append(
                    min(int(record.get("attempt", 0)) + 1 for record in successful)
                )
                if any(
                    not bool(record.get("is_frontier", False))
                    for record in successful
                ):
                    backoff_success_groups.add(group_uid)
        backed_off_groups = set(backoff_depths_by_group)
        depth_drops = [
            max(
                initial_depth_by_group.get(group_uid, 0)
                - min(backoff_depths_by_group[group_uid]),
                0,
            )
            for group_uid in backed_off_groups
        ]
        metrics["prefix_rescue/prefix_backoff_groups"] = float(
            len(backed_off_groups)
        )
        metrics["prefix_rescue/prefix_backoff_success_groups"] = float(
            len(backoff_success_groups)
        )
        metrics["prefix_rescue/prefix_backoff_root_groups"] = float(
            len(root_groups)
        )
        metrics["prefix_rescue/prefix_backoff_depth_drop_mean"] = (
            float(np.mean(depth_drops)) if depth_drops else 0.0
        )
        metrics["prefix_rescue/teacher_success_groups"] = float(
            len(success_groups)
        )
        metrics["prefix_rescue/teacher_success_rate"] = float(
            len(success_groups) / max(len(tries), 1)
        )
        metrics["prefix_rescue/rollouts_until_first_success"] = (
            float(np.mean(first_success_attempts))
            if first_success_attempts else 0.0
        )
        metrics["prefix_rescue/teacher_success_root_groups"] = float(
            len(success_root_groups)
        )
        metrics["prefix_rescue/teacher_success_natural_groups"] = float(
            len(success_natural_groups)
        )
        metrics["prefix_rescue/teacher_success_singleton_groups"] = 0.0
        archived_attempts = sum(
            int(bool(record.get("_artifact_archived", False)))
            for records in attempts_by_group.values()
            for record in records
        )
        metrics["prefix_rescue/attempt_archive_records"] = float(
            archived_attempts
        )
        expected_attempts = int(sum(attempts_used.values()))
        if archived_attempts != expected_attempts:
            raise RuntimeError(
                "prefix rescue attempt archive mismatch before actor update: "
                f"scheduled={expected_attempts}, archived={archived_attempts}"
            )
        if not successes:
            return self._rescue_metric_aliases(metrics), None

        query_protos = [success["teacher_proto"] for success in successes]
        for teacher_proto in query_protos[1:]:
            self._align_cf_role_keys(teacher_proto, query_protos[0])
        selected_query = (
            query_protos[0]
            if len(query_protos) == 1
            else DataProto.concat(query_protos)
        )
        pair_records = []
        for teacher_row, success in enumerate(successes):
            job = success["job"]
            for traj_uid, student_row, student_reward in zip(
                job.matched_traj_uids,
                job.matched_rows,
                job.matched_episode_rewards,
            ):
                utility = (float(success["terminal_reward"]) - float(student_reward)) / success_threshold
                if utility <= 0.0:
                    continue
                pair_records.append({
                    "pair_id": (
                        f"prefix:{success['local_group_uid']}:"
                        f"{success['attempt']}:{traj_uid}"
                    ),
                    "pair_idx": len(pair_records),
                    "group_uid": success["local_group_uid"],
                    "original_group_uid": success["group_uid"],
                    "node_uid": job.node_uid,
                    "teacher_row": int(teacher_row),
                    "student_row": int(student_row),
                    "student_traj_uid": str(traj_uid),
                    **self._prefix_rescue_student_identity(
                        student_batch, int(student_row)
                    ),
                    "divergence_turn": int(job.depth),
                    "utility": float(utility),
                    "teacher_full_success": bool(float(success["terminal_reward"]) >= success_threshold),
                    "teacher_task_id": job.task_id,
                    "student_task_id": job.task_id,
                })
        if not pair_records:
            return self._rescue_metric_aliases(metrics), None
        pair_proto = self._interleave_cf_pair_proto(
            pair_records, role1_proto=selected_query, role2_proto=student_batch
        )
        pair_proto.meta_info["skill_int_pair_source"] = "prefix_rescue"
        pair_proto.meta_info["prefix_rescue_pair_records"] = list(pair_records)
        pair_proto.meta_info["prefix_rescue_matched_student_records"] = int(
            len(pair_records)
        )
        archived_pair_records = 0
        for record in pair_records:
            self._append_artifact("rescue_pair", record)
            archived_pair_records += 1
        pair_proto.meta_info["prefix_rescue_archived_pair_records"] = int(
            archived_pair_records
        )
        if archived_pair_records != len(pair_records):
            raise RuntimeError(
                "prefix rescue pair archive mismatch before actor update: "
                f"pairs={len(pair_records)}, archived={archived_pair_records}"
            )
        suffix_examples = []
        for success in successes:
            suffix_examples.extend(
                {
                    **dict(example),
                    "group_uid": success["local_group_uid"],
                    "original_group_uid": success["group_uid"],
                    "node_uid": success["job"].node_uid,
                    "node_hash": int(
                        hashlib.sha1(
                            str(success["job"].node_uid).encode("utf-8")
                        ).hexdigest()[:15],
                        16,
                    ),
                }
                for example in success["suffix_examples"]
            )
        self._last_skill_int_prefix_rescue_suffix_proto = (
            self._build_prefix_rescue_suffix_proto(student_batch, suffix_examples)
        )
        for example in suffix_examples:
            self._append_artifact("rescue_suffix_kl_example", example)
        metrics["prefix_rescue/suffix_rows_emitted"] = float(len(suffix_examples))
        return self._rescue_metric_aliases(metrics), pair_proto

    def _run_prefix_rescue_rollout(self, student_batch, selected_group_uids):
        """Run skill-conditioned continuations from shared failed-student states.

        A verified first-divergence action completes the zero-contrast group for
        clipped policy-gradient training.  Remaining successful actions are kept
        separately for low-weight skill-to-no-skill conditional KL.
        """
        cfg = self._prefix_rescue_cfg()
        self._last_skill_int_prefix_rescue_suffix_proto = None
        if getattr(self, "_cf_branch_envs", None) is None:
            raise RuntimeError("Missing environment pool for verified prefix replay")
        required = ("alt_input_ids", "alt_attention_mask")
        if any(key not in student_batch.batch.keys() for key in required):
            raise RuntimeError("Verified rescue requires both the student and teacher prompt views")

        teacher_cfg = self._teacher_rollout_cfg() or {}
        success_threshold = float(teacher_cfg.get('success_reward_threshold', 10.0))
        candidate_success_threshold = self._optional_float(
            teacher_cfg.get('candidate_success_reward_threshold', None)
        )
        if candidate_success_threshold is None:
            candidate_success_threshold = success_threshold
        tries, selection_metrics = build_failure_prefix_tries(
            student_batch,
            selected_group_uids,
            success_threshold=candidate_success_threshold,
            min_support=int(cfg.get("min_support", 2)),
            max_depth=cfg.get("max_depth", None),
            max_reward_range=cfg.get("max_reward_range", 1e-8),
            max_groups=int(cfg.get("max_groups_per_step", 8)),
            group_size=int(self.config.env.rollout.n),
        )
        metrics = {f"prefix_rescue/{key}": float(value) for key, value in selection_metrics.items()}
        if not tries:
            return self._rescue_metric_aliases(metrics), None
        frontier_jobs = select_deepest_failure_frontier(tries)
        if not frontier_jobs:
            return self._rescue_metric_aliases(metrics), None
        clusters = [failure_prefix_job_to_cluster(job) for job in frontier_jobs]
        self._dump_prefix_rescue_student_groups(student_batch, clusters)
        return self._run_prefix_rescue_jobs(
            student_batch, tries, frontier_jobs, metrics
        )

    def _run_cf_branch_rollout(self, student_batch):
        """Collect learner-state counterfactual branch pairs.

        Pipeline: Φ-stall select on student states -> query teacher
        one action there (with-skill, no env) -> confirm divergence -> prefix
        replay + branch (teacher vs student action) + continuation on isolated
        pool -> A^cf = γ^n[Φ_T − Φ_S] -> emit cf_pair_proto via the shared
        interleave. Returns (metrics, cf_pair_proto|None).
        """
        if not self._cf_branch_enabled() or getattr(self, "_cf_branch_envs", None) is None:
            return {}, None
        if getattr(self, "_skill_int_running_cf_branch", False):
            return {}, None

        # Base generation meta for branch continuation. vLLM rollout hard-accesses
        # meta_info["eos_token_id"], so _cf_batch_generate MUST inherit the student
        # batch's meta_info (mirrors the state-query path); a bare proto would
        # KeyError inside the worker and silently fall back to no-op actions.
        self._cf_branch_gen_meta = dict(student_batch.meta_info)

        # 1) Φ-stall selection on student states (reuse state-query selection with
        #    selection=phi_stall) -> records with student_row/step/session.
        group_values = np.asarray(student_batch.non_tensor_batch.get("uid", []), dtype=object)
        group_utils = {str(g): 1.0 for g in group_values}  # uniform (not gated by teacher)
        saved_selection = self._state_teacher_query_cfg().get("selection")
        records = self._select_student_state_query_records(student_batch, group_utils, force_selection="phi_stall")
        if not records:
            return {"cf_branch/stall_points": 0.0}, None

        # 2) Query teacher one action at each selected student state (with-skill,
        #    no env) -> role-1 proto (no-skill tensors + with-skill teacher logp).
        prompt_batch = self._build_state_teacher_query_prompt_batch(student_batch, records)
        if prompt_batch is None:
            return {"cf_branch/stall_points": float(len(records)), "cf_branch/no_prompt": 1.0}, None
        padded, pad_size = pad_dataproto_to_divisor(prompt_batch, max(int(self.actor_rollout_wg.world_size), 1))
        try:
            role1 = self.actor_rollout_wg.generate_sequences(padded)
            role1 = unpad_dataproto(role1, pad_size=pad_size)
            role1 = self._attach_state_query_source_context(role1, student_batch, records)
        except Exception as err:
            print(f"[SkillSDTrainer] cf_branch teacher query failed (non-fatal): {err}", flush=True)
            return {"cf_branch/query_failed": 1.0}, None

        # 3) Confirm divergence + run branches on isolated pool.
        self._skill_int_running_cf_branch = True
        try:
            branch_results, kept_rows = self._run_cf_branches(student_batch, role1, records)
        finally:
            self._skill_int_running_cf_branch = False
        counts = getattr(self, "_cf_branch_last_counts", {}) or {}
        metrics = {
            "cf_branch/stall_points": float(len(records)),
            "cf_branch/confirmed_divergences": float(counts.get("confirmed_divergences", len(branch_results))),
            "cf_branch/replay_mismatch": float(counts.get("replay_mismatch", 0)),
            "cf_branch/cont_len_mean": float(counts.get("cont_len_mean", 0.0)),
            "cf_branch/run_to_end_frac": float(counts.get("ran_to_end_frac", 0.0)),
            "cf_branch/skill_text_coverage": float(counts.get("skill_text_coverage", 0.0)),
        }
        print(f"[SkillSDTrainer] cf_branch: stall_points={len(records)} "
              f"confirmed_divergences={counts.get('confirmed_divergences', len(branch_results))} "
              f"replay_mismatch={counts.get('replay_mismatch', 0)} "
              f"cont_len_mean={counts.get('cont_len_mean', 0.0):.2f} "
              f"skill_text_coverage={counts.get('skill_text_coverage', 0.0):.2f}", flush=True)
        if not branch_results:
            metrics.update(self._refresh_skill_attribution_feedback())
            return metrics, None

        # 4) Build pairs (signed A^cf, ε_cf filter) and the interleaved proto.
        cfg = self._cf_branch_cfg()
        pairs = _cf_branch.build_branch_pairs(
            branch_results=branch_results,
            gamma=float(cfg.get("gamma", 0.95)),
            epsilon_cf=float(cfg.get("epsilon_cf", 0.0)),
            a_cf_reduce=str(cfg.get("a_cf_reduce", "max_k")),
        )
        metrics.update(_cf_branch.summarize_branch_pairs(pairs))
        # ε_cf reconvergence drops = survived-replay branches whose max_k A^cf ≤ ε_cf
        # (teacher never beat student across the branch -> coarse-Φ reconvergence).
        metrics["cf_branch/epsilon_cf_dropped"] = float(max(len(branch_results) - len(pairs), 0))
        metrics["cf_branch/pairs_emitted"] = float(len(pairs))
        metrics.update(self._observe_skill_attribution(pairs, student_batch, branch_results))
        # Diagnostic dump of ALL survived-replay branches (incl. ε_cf-dropped) so
        # the yield story is inspectable: emitted pairs only show survivors, but the
        # dropped ones (identical curves -> a_cf=0) are what we need to explain low
        # yield. Records teacher/student Φ curves + a_cf + run-to-end per branch.
        self._dump_cf_branch_diagnostics(branch_results, gamma=float(cfg.get("gamma", 0.95)),
                                         a_cf_reduce=str(cfg.get("a_cf_reduce", "max_k")))
        if not pairs:
            return metrics, None

        pair_records = self._cf_branch_pair_records(pairs, role1, kept_rows)
        no_skill_role1 = self._make_no_skill_logprob_proto(role1)
        if no_skill_role1 is None:
            metrics["cf_branch/no_skill_context_failed"] = 1.0
            return metrics, None
        cf_pair_proto = self._interleave_cf_pair_proto(
            pair_records,
            role1_proto=no_skill_role1,
            role2_proto=student_batch,
        )
        self._dump_cf_branch_pairs(pairs)
        print(f"[SkillSDTrainer] cf_branch: emitted {len(pairs)} branch pairs "
              f"(teacher_pref={sum(1 for p in pairs if p.a_cf>0)}, student_pref={sum(1 for p in pairs if p.a_cf<0)})", flush=True)
        return metrics, cf_pair_proto

    def _cf_batch_generate(self, obs_texts, avail_lists, task_descs, roles=None, skill_texts=None):
        """Sample one action per live slot using its skill or skill-free context."""
        n = len(obs_texts)
        if n == 0:
            return []
        env = getattr(self, "_cf_branch_env", "webshop")
        fallback = self._cf_placeholder_action()
        roles = list(roles) if roles is not None else ["student"] * n
        skill_texts = list(skill_texts) if skill_texts is not None else [""] * n
        prompts = []
        no_skill_prompts = []
        for i in range(n):
            td, avail = str(task_descs[i] or ""), (avail_lists[i] or [])
            obs_context = obs_texts[i]
            if isinstance(obs_context, dict) and "launch_prompts" in obs_context:
                student_text, teacher_text = obs_context["launch_prompts"]
                if not student_text or not teacher_text:
                    raise ValueError("Missing student or teacher context at the restored prefix")
                no_skill_prompts.append(student_text)
                prompts.append(teacher_text)
                continue
            if isinstance(obs_context, dict):
                obs = str(obs_context.get("observation", "") or "")
                step_count = int(obs_context.get("step_count", 0) or 0)
                history_length = int(obs_context.get("history_length", 0) or 0)
                action_history = str(obs_context.get("action_history", "") or "")
            else:
                obs = str(obs_context or "")
                step_count = 0
                history_length = 0
                action_history = ""
            prompt_kwargs = {
                "env": env,
                "step_count": step_count,
                "history_length": history_length,
                "action_history": action_history,
            }
            no_skill_prompt = _cf_branch.build_no_skill_prompt(
                td, obs, avail, **prompt_kwargs
            )
            no_skill_prompts.append(no_skill_prompt)
            if str(roles[i]) == "teacher":
                prompts.append(_cf_branch.build_with_skill_prompt(
                    td,
                    obs,
                    avail,
                    str(skill_texts[i] or ""),
                    **prompt_kwargs,
                ))
            else:
                prompts.append(no_skill_prompt)
        try:
            template_kwargs = self.config.data.get(
                "apply_chat_template_kwargs", {}
            )
            def encode_prompts(texts):
                rendered = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], add_generation_prompt=True,
                        tokenize=False, **template_kwargs,
                    )
                    for prompt in texts
                ]
                return self.tokenizer(
                    rendered, return_tensors="pt", padding=True, truncation=True,
                    max_length=int(self.config.data.max_prompt_length), add_special_tokens=False,
                )

            prev_side = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = "left"
                enc = encode_prompts(prompts)
                student_enc = encode_prompts(no_skill_prompts)
            finally:
                self.tokenizer.padding_side = prev_side
            from verl import DataProto as _DP
            attn = enc["attention_mask"]
            proto = _DP.from_dict(tensors={
                "input_ids": enc["input_ids"], "attention_mask": attn,
                "position_ids": attn.long().cumsum(-1) - 1,
            })
            meta = dict(getattr(self, "_cf_branch_gen_meta", {}) or {})
            rollout_cfg = (
                self._prefix_rescue_cfg()
                if getattr(self, "_skill_int_running_prefix_rescue", False)
                else self._cf_branch_cfg()
            )
            do_sample = bool(rollout_cfg.get("continuation_do_sample", False))
            meta["do_sample"] = do_sample
            # n=1 is required because branch slots already represent independent
            # continuations; expanding each slot by rollout.n breaks env alignment.
            meta["validate"] = False
            proto.meta_info.update(meta)
            padded, pad = pad_dataproto_to_divisor(proto, max(int(self.actor_rollout_wg.world_size), 1))
            out = unpad_dataproto(self.actor_rollout_wg.generate_sequences(padded), pad_size=pad)
            resp_rows = out.batch["responses"]
            # Default (greedy, do_sample=False) yields 1 row per prompt. If sampling
            # expanded rows (n>1), take the first sample per prompt via stride so
            # indexing stays 1:1 with the inputs.
            stride = max(len(resp_rows) // max(n, 1), 1)
            actions = []
            examples = []
            for i in range(n):
                ri = min(i * stride, len(resp_rows) - 1)
                resp = self.tokenizer.decode(resp_rows[ri], skip_special_tokens=True)
                projected_action = _cf_branch.extract_action_text(
                    resp,
                    env=env,
                    available_actions=(avail_lists[i] or [])
                    if env == "alfworld" else None,
                )
                action_valid = projected_action is not None
                action = projected_action or fallback
                actions.append(action)
                examples.append({
                    "prompt": no_skill_prompts[i],
                    "no_skill_prompt": no_skill_prompts[i],
                    "skill_prompt": prompts[i],
                    "student_prompt_ids": student_enc["input_ids"][i][student_enc["attention_mask"][i].bool()].tolist(),
                    "teacher_prompt_ids": enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist(),
                    "response_ids": resp_rows[ri][out.batch["attention_mask"][ri, -resp_rows.size(1):].bool()].tolist(),
                    "response": resp,
                    "action": action,
                    "action_valid": bool(action_valid),
                    "available_action_count": int(len(avail_lists[i] or [])),
                    "role": str(roles[i]),
                    "environment_step": int(
                        obs_texts[i].get("step_count", 0)
                        if isinstance(obs_texts[i], dict) else 0
                    ),
                })
            self._cf_last_generation_examples = examples
            return actions
        except Exception as err:
            print(f"[SkillSDTrainer] cf_branch batched continuation failed (non-fatal): {err}", flush=True)
            self._cf_last_generation_examples = [
                {
                    "prompt": no_skill_prompts[i],
                    "no_skill_prompt": no_skill_prompts[i],
                    "skill_prompt": prompts[i],
                    "response": "",
                    "action": fallback,
                    "action_valid": False,
                    "available_action_count": int(len(avail_lists[i] or [])),
                    "role": str(roles[i]),
                    "generation_error": str(err),
                }
                for i in range(n)
            ]
            return [fallback] * n

    def _run_branches_breadth_first(self, specs, *, n_cont, max_steps, batch_generate=None):
        """Breadth-first multi-step branch continuation on the isolated pool.

        ``specs`` is a list of per-branch dicts:
          {session:int, script:list[str], prefix_len:int, task_desc:str}
        where script = prefix_actions + [first_branch_action]. Each branch gets its
        OWN persistent worker slot and is NEVER reset mid-flight (env state = session
        + action history persists). The loop mirrors the main rollout loop: at each
        step, slots still inside ``script`` replay a deterministic action; slots past
        it enter the "generate phase" and their next action comes from ONE top-level
        batched generate call (``batch_generate`` -> list[str]); done slots get a
        placeholder and are masked. This keeps generate_sequences at top-level call
        depth.

        Returns a list aligned to ``specs``: each is a dict with
          {phi_curve:list[float], phi_end:float, n:int, done:bool, mismatch:bool}
        phi_curve is recorded from the divergence step onward (t >= prefix_len), so
        it has up to 1 + n_cont points (fewer if the episode ended early).
        Chunks specs into groups of <= pool_n slots and runs each independently.
        ``batch_generate`` defaults to self._cf_batch_generate (injectable for tests).
        """
        from recipe.skill_sd import phi_compiler as _phi
        if batch_generate is None:
            batch_generate = self._cf_batch_generate
        raw = self._cf_branch_envs
        pool_n = self._branch_replay_pool_size()
        results = [None] * len(specs)
        # Chunk so each pass fits the pool; pairing happens later by record_idx, so a
        # teacher/student pair may span chunks without harm (curves are independent).
        for start in range(0, len(specs), max(pool_n, 1)):
            chunk = list(range(start, min(start + max(pool_n, 1), len(specs))))
            self._run_branch_chunk(chunk, specs, results, raw=raw, n_cont=n_cont,
                                   max_steps=max_steps, batch_generate=batch_generate, phi=_phi)
        return results

    @staticmethod
    def _search_branch_history(obs_hist, acts_hist) -> str:
        entries = []
        for step, action in enumerate(acts_hist, start=1):
            entries.append(f"Step {step} action: {action}")
            if step < len(obs_hist) and str(obs_hist[step] or "").strip():
                entries.append(f"Step {step} information: {obs_hist[step]}")
        return "\n".join(entries) or "(no prior search interaction)"

    @staticmethod
    def _alfworld_branch_context(obs_hist, acts_hist, history_limit: int) -> dict:
        """Format branch state exactly like ALFWorldEnvironmentManager memory."""
        step_count = min(len(acts_hist), max(len(obs_hist) - 1, 0))
        limit = max(int(history_limit), 0)
        start = max(0, step_count - limit)
        lines = [
            f"[Observation {idx + 1}: '{obs_hist[idx]}', Action {idx + 1}: '{acts_hist[idx]}']"
            for idx in range(start, step_count)
        ]
        return {
            "observation": str(obs_hist[-1] if obs_hist else ""),
            "step_count": step_count,
            "history_length": step_count - start,
            "action_history": "\n".join(lines),
        }

    def _run_branch_chunk(self, chunk, specs, results, *, raw, n_cont, max_steps, batch_generate, phi):
        """One breadth-first lockstep pass over a chunk of branch slots (<= pool_n)."""
        env = getattr(self, "_cf_branch_env", "webshop")
        # WebShop sessions are int goal ids; ALFWorld sessions are gamefile path
        # strings (the adapter handles reset arity/kwargs).
        sessions = [specs[i]["session"] for i in chunk]
        milestones = [
            self._cf_compile_milestone(specs[i])
            if bool(specs[i].get("track_phi", True))
            else None
            for i in chunk
        ]
        placeholder = self._cf_placeholder_action()
        try:
            obs_list, info_list = self._cf_reset_pool(sessions)
        except Exception as err:
            print(f"[SkillSDTrainer] cf_branch chunk reset failed (non-fatal): {err}", flush=True)
            for i in chunk:
                results[i] = {"phi_curve": [], "phi_end": 0.0, "n": 0, "done": False, "mismatch": True}
            return
        m = len(chunk)
        obs_hist = [[obs_list[j]] for j in range(m)]
        acts_hist = [[] for _ in range(m)]
        infos = [info_list[j] for j in range(m)]
        done = [False] * m
        mismatch = [False] * m
        replay_checks = [0] * m
        curves = [[] for _ in range(m)]
        no_skill_suffix_examples = [[] for _ in range(m)]
        generation_examples_by_turn = [{} for _ in range(m)]
        generation_failed = [False] * m
        generation_failure_turn = [None] * m
        suffix_generation_count = [0] * m
        suffix_valid_action_count = [0] * m
        suffix_invalid_generation_count = [0] * m
        suffix_retry_count = [0] * m
        reward_hist = [[] for _ in range(m)]
        done_hist = [[] for _ in range(m)]
        task_score_hist = [[] for _ in range(m)]
        last_reward = [0.0] * m
        horizons = [min(specs[chunk[j]]["prefix_len"] + 1 + int(n_cont), max_steps) for j in range(m)]
        prefix_lens = [int(specs[chunk[j]]["prefix_len"]) for j in range(m)]
        scripts = [list(specs[chunk[j]]["script"]) for j in range(m)]
        expected_observations = [
            tuple(specs[chunk[j]].get("expected_observations", ()) or ())
            for j in range(m)
        ]
        alfworld_history_limit = 4
        if env == "alfworld":
            try:
                alfworld_history_limit = int(getattr(self.config.env, "history_length", 4))
            except (AttributeError, TypeError, ValueError):
                pass
        continuation_cfg = (
            self._prefix_rescue_cfg()
            if getattr(self, "_skill_int_running_prefix_rescue", False)
            else self._cf_branch_cfg()
        )
        invalid_action_retries = (
            max(int(continuation_cfg.get("continuation_invalid_action_retries", 0)), 0)
            if env == "alfworld" else 0
        )
        for j in range(m):
            if not expected_observations[j]:
                continue
            replay_checks[j] += 1
            actual = " ".join(str(obs_list[j] or "").strip().lower().split())
            expected = " ".join(str(expected_observations[j][0] or "").strip().lower().split())
            if not actual or actual != expected:
                mismatch[j] = True
        max_t = max(horizons) if horizons else 0
        for t in range(max_t):
            # Decide each slot's action: scripted (t < len(script)) or generate phase.
            gen_slots, gen_obs, gen_avail, gen_task, gen_role, gen_skill = [], [], [], [], [], []
            for j in range(m):
                if done[j] or mismatch[j] or generation_failed[j] or t >= horizons[j]:
                    continue
                if t >= len(scripts[j]):
                    info = infos[j] if isinstance(infos[j], dict) else {}
                    obs_txt, avail = self._cf_extract_obs_avail(info)
                    if env == "searchqa":
                        obs_txt = self._search_branch_history(obs_hist[j], acts_hist[j])
                    elif env == "alfworld":
                        obs_txt = self._alfworld_branch_context(
                            obs_hist[j], acts_hist[j], alfworld_history_limit
                        )
                        spec = specs[chunk[j]]
                        if t == prefix_lens[j] and "launch_prompts" in spec:
                            obs_txt["launch_prompts"] = spec["launch_prompts"]
                        obs_txt.update({
                            "task_id": str(
                                spec.get("task_id")
                                or spec.get("session")
                                or ""
                            ),
                            "actions": tuple(acts_hist[j]),
                            "observations": tuple(obs_hist[j]),
                        })
                    gen_slots.append(j)
                    gen_obs.append(obs_txt)
                    gen_avail.append(avail)
                    gen_task.append(str(specs[chunk[j]].get("task_desc", "") or ""))
                    # OPSD: teacher slots continue WITH the task skill text, student
                    # slots WITHOUT (see _cf_batch_generate).
                    gen_role.append(str(specs[chunk[j]].get("role", "student")))
                    gen_skill.append(str(specs[chunk[j]].get("skill_text", "") or ""))
            input_index_by_slot = {
                slot: index for index, slot in enumerate(gen_slots)
            }
            generation_attempts_by_slot = {slot: [] for slot in gen_slots}
            gen_map = {}
            pending_slots = list(gen_slots)
            for retry_index in range(invalid_action_retries + 1):
                if not pending_slots:
                    break
                input_indices = [input_index_by_slot[slot] for slot in pending_slots]
                self._cf_last_generation_examples = []
                retry_actions = batch_generate(
                    [gen_obs[index] for index in input_indices],
                    [gen_avail[index] for index in input_indices],
                    [gen_task[index] for index in input_indices],
                    roles=[gen_role[index] for index in input_indices],
                    skill_texts=[gen_skill[index] for index in input_indices],
                )
                generated_examples = list(
                    getattr(self, "_cf_last_generation_examples", []) or []
                )
                next_pending = []
                for response_index, slot in enumerate(pending_slots):
                    input_index = input_index_by_slot[slot]
                    action = (
                        retry_actions[response_index]
                        if response_index < len(retry_actions) else placeholder
                    )
                    example = (
                        dict(generated_examples[response_index])
                        if response_index < len(generated_examples) else {
                            "action": action,
                            "response": "",
                        }
                    )
                    if "action_valid" in example:
                        action_valid = bool(example["action_valid"])
                    elif env == "alfworld":
                        projected = _cf_branch.extract_action_text(
                            action,
                            env="alfworld",
                            available_actions=gen_avail[input_index] or [],
                        )
                        action_valid = projected is not None
                        if projected is not None:
                            action = projected
                    else:
                        action_valid = bool(str(action or "").strip())
                    example.update({
                        "action": action,
                        "action_valid": bool(action_valid),
                        "retry_index": int(retry_index),
                    })
                    generation_attempts_by_slot[slot].append(example)
                    suffix_generation_count[slot] += 1
                    if retry_index > 0:
                        suffix_retry_count[slot] += 1
                    if action_valid:
                        suffix_valid_action_count[slot] += 1
                        gen_map[slot] = action
                    else:
                        suffix_invalid_generation_count[slot] += 1
                        next_pending.append(slot)
                pending_slots = next_pending

            for slot in gen_slots:
                attempts = generation_attempts_by_slot[slot]
                if not attempts:
                    continue
                final_example = dict(attempts[-1])
                final_example["generation_attempts"] = attempts
                final_example["retry_count"] = max(len(attempts) - 1, 0)
                generation_examples_by_turn[slot][t] = final_example
                if str(specs[chunk[slot]].get("role", "student")) == "teacher":
                    no_skill_suffix_examples[slot].append(final_example)
                if slot not in gen_map:
                    generation_failed[slot] = True
                    generation_failure_turn[slot] = int(t)
            # Build the full action list for ALL slots (env requires len == m).
            actions = []
            for j in range(m):
                if done[j] or mismatch[j] or generation_failed[j] or t >= horizons[j]:
                    actions.append(placeholder)  # placeholder, result ignored
                elif t < len(scripts[j]):
                    actions.append(scripts[j][t])
                else:
                    actions.append(gen_map.get(j, placeholder))
            o, rr, dd, ii = self._cf_step_pool(actions)
            for j in range(m):
                if done[j] or mismatch[j] or generation_failed[j] or t >= horizons[j]:
                    continue
                obs_hist[j].append(o[j]); acts_hist[j].append(actions[j]); infos[j] = ii[j]
                last_reward[j] = float(rr[j])
                is_done = bool(dd[j] if not isinstance(dd[j], (list, tuple)) else dd[j][0])
                reward_hist[j].append(float(rr[j]))
                done_hist[j].append(is_done)
                step_info = ii[j] if isinstance(ii[j], dict) else {}
                step_task_score = 0.0
                for score_key in (
                    "episode_task_score", "task_score", "taskscore", "score"
                ):
                    if score_key in step_info:
                        step_task_score = self._float_value(
                            step_info.get(score_key), 0.0
                        )
                        break
                task_score_hist[j].append(step_task_score)
                if t < prefix_lens[j]:
                    expected_states = expected_observations[j]
                    if expected_states:
                        replay_checks[j] += 1
                        expected_idx = t + 1
                        actual = " ".join(str(o[j] or "").strip().lower().split())
                        expected = (
                            " ".join(str(expected_states[expected_idx] or "").strip().lower().split())
                            if expected_idx < len(expected_states)
                            else ""
                        )
                        if not expected or actual != expected:
                            mismatch[j] = True
                    if is_done:
                        mismatch[j] = True  # prefix ended early -> replay mismatch (drop)
                    continue
                # t >= prefix_len: divergence step onward -> record Φ.
                if milestones[j] is None:
                    curves[j].append(0.0)
                else:
                    curves[j].append(phi.phi_value(
                        env,
                        milestones[j],
                        obs_hist[j],
                        acts_hist[j],
                        terminal_reward=last_reward[j] if is_done else None,
                    ))
                if is_done:
                    done[j] = True
        for j in range(m):
            i = chunk[j]
            final_info = infos[j] if isinstance(infos[j], dict) else {}
            terminal_task_score = 0.0
            for score_key in (
                "episode_task_score",
                "task_score",
                "taskscore",
                "score",
            ):
                if score_key in final_info:
                    terminal_task_score = self._float_value(
                        final_info.get(score_key), 0.0
                    )
                    break
            step_records = []
            for turn, action in enumerate(acts_hist[j]):
                generation = generation_examples_by_turn[j].get(turn, {})
                step_records.append({
                    "turn": int(turn),
                    "observation": obs_hist[j][turn]
                    if turn < len(obs_hist[j]) else None,
                    "skill_prompt": generation.get("skill_prompt"),
                    "no_skill_prompt": generation.get("no_skill_prompt"),
                    "raw_response": generation.get("response"),
                    "parsed_action": action,
                    "generation_action_valid": generation.get("action_valid"),
                    "generation_attempts": generation.get("generation_attempts"),
                    "next_observation": obs_hist[j][turn + 1]
                    if turn + 1 < len(obs_hist[j]) else None,
                    "reward": reward_hist[j][turn]
                    if turn < len(reward_hist[j]) else 0.0,
                    "task_score": task_score_hist[j][turn]
                    if turn < len(task_score_hist[j]) else 0.0,
                    "done": done_hist[j][turn]
                    if turn < len(done_hist[j]) else False,
                })
            if generation_failed[j] and generation_failure_turn[j] is not None:
                failure_turn = int(generation_failure_turn[j])
                generation = generation_examples_by_turn[j].get(failure_turn, {})
                step_records.append({
                    "turn": failure_turn,
                    "observation": obs_hist[j][-1] if obs_hist[j] else None,
                    "skill_prompt": generation.get("skill_prompt"),
                    "no_skill_prompt": generation.get("no_skill_prompt"),
                    "raw_response": generation.get("response"),
                    "parsed_action": "",
                    "generation_action_valid": False,
                    "generation_attempts": generation.get("generation_attempts"),
                    "next_observation": None,
                    "reward": 0.0,
                    "task_score": 0.0,
                    "done": False,
                    "failure_reason": "no_admissible_teacher_action",
                })
            results[i] = {
                "phi_curve": curves[j],
                "phi_end": (curves[j][-1] if curves[j] else 0.0),
                "n": len(curves[j]),
                "done": done[j],
                "mismatch": mismatch[j],
                "replay_verified": bool(expected_observations[j]) and not mismatch[j]
                and replay_checks[j] == prefix_lens[j] + 1,
                "terminal_reward": float(last_reward[j]) if done[j] else 0.0,
                "terminal_task_score": terminal_task_score,
                "generation_failed": bool(generation_failed[j]),
                "generation_failure_turn": generation_failure_turn[j],
                "suffix_generation_count": int(suffix_generation_count[j]),
                "suffix_valid_action_count": int(suffix_valid_action_count[j]),
                "suffix_invalid_generation_count": int(
                    suffix_invalid_generation_count[j]
                ),
                "suffix_retry_count": int(suffix_retry_count[j]),
                "actions": list(acts_hist[j]),
                "observations": list(obs_hist[j]),
                "steps": step_records,
                "no_skill_suffix_examples": list(no_skill_suffix_examples[j]),
            }

    def _cf_opsd_scores(self, role1):
        """Per-row OPSD/SDAR gate signal at the divergence state:
        S = logπ(a_teacher | with-skill) − logπ(a_teacher | no-skill), the exact
        quantity score-based OPSD (SDAR gate) uses to decide/scale distillation.

        role1.input_ids = with-skill prompt + teacher action; role1.alt_input_ids
        = no-skill prompt (attached by _attach_state_query_source_context). We
        contrast the teacher action's mean-token logp under both views. Returned as
        an array with one value per row, or None if unavailable. This metric is logged only.
        """
        try:
            if role1 is None or "alt_input_ids" not in role1.batch.keys():
                return None
            with_skill_lp = self._compute_log_probs_for_proto(role1).detach().cpu()
            no_skill_proto = self._make_no_skill_logprob_proto(role1)
            if no_skill_proto is None:
                return None
            no_skill_lp = self._compute_log_probs_for_proto(no_skill_proto).detach().cpu()
            response_len = int(role1.batch["responses"].shape[-1])
            resp_mask = role1.batch["attention_mask"][:, -response_len:].detach().cpu()
            mask = resp_mask.to(with_skill_lp.dtype)
            denom = mask.sum(dim=-1).clamp(min=1.0)
            row_with = (with_skill_lp * mask).sum(dim=-1) / denom
            row_no = (no_skill_lp * mask).sum(dim=-1) / denom
            return (row_with - row_no).numpy()
        except Exception as err:
            print(f"[SkillSDTrainer] cf_branch OPSD-score computation failed (non-fatal): {err}", flush=True)
            return None

    def _run_cf_branches(self, student_batch, role1, records):
        """Confirm divergence and execute prefix-replay branches on the isolated
        pool. Returns (branch_results, kept_role1_rows) where branch_results feeds
        cf_branch.build_branch_pairs and kept_role1_rows maps to role1 rows."""
        cfg = self._cf_branch_cfg()
        n_cont = int(cfg.get("n_continuation", 4))
        max_steps = int(getattr(self.config.env, "max_steps", 15))
        sfields = student_batch.non_tensor_batch

        # Pass 1: confirm divergence, reconstruct prefix, build TWO branch specs
        # (teacher-action, student-action) per kept record. specs are flattened;
        # spec_pairs[k] = (teacher_spec_idx, student_spec_idx) for kept record k.
        # Log the context-dependent score at each divergence.
        opsd_scores = self._cf_opsd_scores(role1)

        specs = []
        spec_pairs = []
        kept_meta = []
        kept_rows = []
        for r1_idx, rec in enumerate(records):
            srow = int(rec["student_row"])
            stall_step = int(rec.get("student_step", 0))
            _cf_env = getattr(self, "_cf_branch_env", "webshop")
            teacher_action = _cf_branch.extract_action_text(
                self._field_at(role1.non_tensor_batch, "model_response", r1_idx, "")
                or (self.tokenizer.decode(role1.batch["responses"][r1_idx], skip_special_tokens=True)
                    if "responses" in role1.batch else ""),
                env=_cf_env)
            student_action = _cf_branch.extract_action_text(
                self._field_at(sfields, "model_response", srow, ""), env=_cf_env)
            if not _cf_branch.actions_diverge(teacher_action, student_action):
                continue
            # Reconstruct prefix = student's own actions before the stall step.
            tuid = str(self._field_at(sfields, "traj_uid", srow, ""))
            traj_rows = [i for i in range(len(sfields.get("traj_uid", []))) if str(sfields["traj_uid"][i]) == tuid]
            traj_rows.sort(key=lambda i: self._int_value(self._field_at(sfields, "step_idx", i), i))
            step_actions = [_cf_branch.extract_action_text(self._field_at(sfields, "model_response", i, ""), env=_cf_env) or "" for i in traj_rows]
            try:
                stall_pos = traj_rows.index(srow)
            except ValueError:
                stall_pos = min(stall_step, len(step_actions))
            prefix = step_actions[:stall_pos]
            if getattr(self, "_cf_branch_env", "webshop") == "alfworld":
                # ALFWorld session = gamefile path string (reset(gamefiles=...)); keep
                # it raw, no int-cast. Prefer extra.gamefile, fall back to task_id.
                session = self._field_at(sfields, "extra.gamefile", srow, None) \
                    or self._field_at(sfields, "task_id", srow, None)
                if session is None:
                    continue
                session = str(session)
            else:
                session = self._field_at(sfields, "session", srow, None) or self._field_at(sfields, "task_id", srow, None)
                if session is None:
                    continue
                try:
                    session = int(str(session).split(":")[-1])
                except (ValueError, TypeError):
                    continue
            task_desc = str(self._field_at(sfields, "instruction_text", srow, "") or "")
            if _cf_env == "alfworld" and not task_desc:
                task_desc = self._alfworld_task_description(sfields, srow)
            # OPSD: teacher branch continues WITH the task's skill text (extracted
            # from the with-skill alt-prompt view; task-level static). Student branch
            # continues WITHOUT. This makes A^cf measure the skill's information value
            # over the whole continuation, not just the first (reconverging) step.
            skill_text = _cf_branch.extract_skill_text(
                self._field_at(sfields, "alt_prompt_text", srow, ""))
            t_idx = len(specs)
            specs.append({"session": session, "script": list(prefix) + [teacher_action],
                          "prefix_len": len(prefix), "task_desc": task_desc,
                          "role": "teacher", "skill_text": skill_text})
            s_idx = len(specs)
            specs.append({"session": session, "script": list(prefix) + [student_action],
                          "prefix_len": len(prefix), "task_desc": task_desc,
                          "role": "student", "skill_text": ""})
            spec_pairs.append((t_idx, s_idx))
            opsd_score = (float(opsd_scores[r1_idx])
                          if opsd_scores is not None and r1_idx < len(opsd_scores)
                          else None)
            meta = {"group_uid": str(self._field_at(sfields, "uid", srow, "")),
                    "session": str(session), "student_traj_uid": tuid,
                    "student_row": srow, "stall_step": stall_step,
                    "teacher_action": teacher_action, "student_action": student_action,
                    "prefix_actions": tuple(prefix), "opsd_score": opsd_score}
            if self._skill_attribution_enabled():
                meta["task_type"] = self._skill_attribution_task_type(task_desc)
            kept_meta.append(meta)
            kept_rows.append(r1_idx)

        if not specs:
            return [], []

        # Pass 2: breadth-first multi-step continuation on the isolated pool.
        branch_out = self._run_branches_breadth_first(specs, n_cont=n_cont, max_steps=max_steps)

        # Pass 3: pair teacher/student per kept record -> branch_results. Drop pairs
        # where either branch hit a replay mismatch (prefix ended early).
        results = []
        final_rows = []
        n_mismatch = 0
        cont_lens = []
        ran_to_end = 0
        for k, (t_idx, s_idx) in enumerate(spec_pairs):
            t, s = branch_out[t_idx], branch_out[s_idx]
            if t is None or s is None or t["mismatch"] or s["mismatch"]:
                n_mismatch += 1
                continue
            branch = {"phi_teacher_end": t["phi_end"], "phi_student_end": s["phi_end"],
                      "phi_teacher_curve": t["phi_curve"], "phi_student_curve": s["phi_curve"],
                      "n_continuation": max(min(t["n"], s["n"]), 1),
                      "ran_to_end": bool(t["done"] or s["done"]),
                      "teacher_ran_to_end": bool(t["done"]), "student_ran_to_end": bool(s["done"])}
            branch.update(kept_meta[k])
            results.append(branch)
            final_rows.append(kept_rows[k])
            cont_lens.append(max(min(t["n"], s["n"]), 1))
            ran_to_end += int(bool(t["done"] or s["done"]))
        # Skill-text coverage: fraction of teacher specs that actually got a
        # non-empty skill block. If low, teacher branch silently falls back to
        # no-skill continuation -> reconvergence -> low yield (must be observable).
        n_teacher = sum(1 for sp in specs if str(sp.get("role")) == "teacher")
        n_skill = sum(1 for sp in specs if str(sp.get("role")) == "teacher"
                      and str(sp.get("skill_text", "") or "").strip())
        # Drop-reason accounting (ε_cf drops counted by caller).
        self._cf_branch_last_counts = {
            "confirmed_divergences": len(spec_pairs),
            "replay_mismatch": n_mismatch,
            "branch_results": len(results),
            "cont_len_mean": (float(np.mean(cont_lens)) if cont_lens else 0.0),
            "ran_to_end_frac": (ran_to_end / len(results) if results else 0.0),
            "skill_text_coverage": (n_skill / n_teacher if n_teacher else 0.0),
        }
        return results, final_rows

    def _cf_branch_pair_records(self, pairs, role1, kept_rows):
        """Records for _interleave_cf_pair_proto: teacher_row -> role1 index,
        student_row -> student_batch index, utility = A^cf, pair_idx sequential."""
        records = []
        for pair_idx, (pair, r1_idx) in enumerate(zip(pairs, kept_rows)):
            records.append({
                "pair_id": pair.pair_id,
                "pair_idx": pair_idx,
                "teacher_row": int(r1_idx),          # into role1 (state-query proto)
                "student_row": int(pair.student_row),  # into student_batch
                "utility": float(pair.a_cf),
                "student_traj_uid": pair.student_traj_uid,
                "divergence_turn": int(pair.stall_step),
                "teacher_task_id": None,
                "student_task_id": None,
            })
        return records

    def _dump_cf_branch_pairs(self, pairs) -> None:
        path = self._resolve_span_dump_dir()
        if not path or not pairs:
            return
        import os as _os
        _os.makedirs(path, exist_ok=True)
        fp = _os.path.join(path, f"step_{int(getattr(self, 'global_steps', 0)):06d}_cf_branch_pairs.jsonl")
        self._append_jsonl(fp, {"pairs": [p.__dict__ for p in pairs]})

    def _dump_cf_branch_diagnostics(self, branch_results, *, gamma, a_cf_reduce) -> None:
        """Dump EVERY survived-replay branch (incl. ε_cf-dropped) with its Φ curves
        and computed a_cf, so low yield is explainable (dropped pairs have identical
        curves -> a_cf=0; this shows whether that's reconvergence, plateau, or a
        fallback-action collapse). Diagnostic-only; no effect on training."""
        path = self._resolve_span_dump_dir()
        if not path or not branch_results:
            return
        import os as _os
        _os.makedirs(path, exist_ok=True)
        fp = _os.path.join(path, f"step_{int(getattr(self, 'global_steps', 0)):06d}_cf_branch_diagnostics.jsonl")
        rows = []
        for r in branch_results:
            a_cf = _cf_branch.compute_a_cf(
                phi_teacher_end=r["phi_teacher_end"], phi_student_end=r["phi_student_end"],
                gamma=gamma, n=r.get("n_continuation", 1), reduce=a_cf_reduce,
                phi_teacher_curve=r.get("phi_teacher_curve"),
                phi_student_curve=r.get("phi_student_curve"))
            rows.append({
                "stall_step": r.get("stall_step"), "a_cf": float(a_cf),
                "opsd_score": r.get("opsd_score"),
                "teacher_action": r.get("teacher_action"), "student_action": r.get("student_action"),
                "phi_teacher_curve": r.get("phi_teacher_curve"),
                "phi_student_curve": r.get("phi_student_curve"),
                "phi_teacher_end": r.get("phi_teacher_end"), "phi_student_end": r.get("phi_student_end"),
                "n_continuation": r.get("n_continuation"),
                "teacher_ran_to_end": r.get("teacher_ran_to_end"),
                "student_ran_to_end": r.get("student_ran_to_end"),
            })
        self._append_jsonl(fp, {"branches": rows})

    def _decision_teacher_traj_pool(self, decision):
        eligible_trajs = decision.get("successful_teacher_traj_uids", ()) or ()
        if isinstance(eligible_trajs, str):
            eligible_trajs = (eligible_trajs,)
        pool = {str(value) for value in eligible_trajs}
        if not pool and decision.get("best_teacher_traj_uid") is not None:
            pool = {str(decision.get("best_teacher_traj_uid"))}
        return pool

    def _teacher_aux_active_indices(self, teacher_batch):
        decisions = teacher_batch.meta_info.get("skill_int_teacher_decisions", {}) or {}
        if not decisions:
            return [], []
        teacher_cfg = self._teacher_rollout_cfg() or {}
        aux_gate_mode = str(teacher_cfg.get("aux_gate_mode", "success"))
        utility_eps = float(teacher_cfg.get("utility_eps", 0.0))
        use_group_utility = bool(teacher_cfg.get("aux_use_group_utility", True))
        fields = teacher_batch.non_tensor_batch
        group_values = np.asarray(fields.get("uid", []), dtype=object)
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        # L_suffix demo rows = teacher turns AFTER t* only (marked by
        # _annotate_teacher_cf_span_pairs), NOT the whole successful trajectory.
        rescue_mode = bool(self.skill_sd_cfg.get("rescue_pairing", False))
        suffix_row_mask = fields.get("skill_int_rescue_suffix_row", None)
        active_indices = []
        utilities = []
        for i, group_uid_raw in enumerate(group_values):
            group_uid = str(group_uid_raw)
            decision = decisions.get(group_uid)
            if decision is None:
                continue
            teacher_success = bool(decision.get("teacher_success", False))
            utility = float(decision.get("utility", 0.0))
            traj_uid = str(traj_values[i]) if len(traj_values) else None
            if not _teacher_aux_gate_active(aux_gate_mode, teacher_success, utility, utility_eps):
                continue
            if rescue_mode:
                if suffix_row_mask is None or not bool(suffix_row_mask[i]):
                    continue
            elif aux_gate_mode != "continuous":
                eligible_trajs = self._decision_teacher_traj_pool(decision)
                if eligible_trajs and traj_uid not in eligible_trajs:
                    continue
            active_indices.append(i)
            utilities.append(utility if use_group_utility else 0.0)
        return active_indices, utilities

    def _teacher_aux_quality_weights(self, teacher_batch) -> torch.Tensor:
        cfg = self._teacher_rollout_cfg() or {}
        mode = str(cfg.get("aux_weight_mode", "group_utility"))
        n_rows = len(teacher_batch.batch)
        if mode in {"group_utility", "uniform"}:
            return torch.ones(n_rows, dtype=torch.float32)
        if mode != "trajectory_quality":
            raise ValueError(f"Unknown teacher aux_weight_mode: {mode}")

        quality_floor = float(np.clip(cfg.get("aux_quality_floor", 0.0), 0.0, 1.0))
        success_threshold = max(float(cfg.get("success_reward_threshold", 10.0)), 1e-8)
        query_qualities = teacher_batch.meta_info.get("skill_int_state_query_qualities", None)
        if query_qualities is not None and len(query_qualities) == n_rows:
            raw = [float(value) for value in query_qualities]
        else:
            fields = teacher_batch.non_tensor_batch
            raw = []
            for row in range(n_rows):
                task_score = self._float_value(self._field_at(fields, "episode_task_scores", row, 0.0))
                reward = self._float_value(self._field_at(fields, "episode_rewards", row, 0.0))
                raw.append(max(task_score, reward / success_threshold))
        quality = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
        quality = quality_floor + (1.0 - quality_floor) * quality
        return torch.as_tensor(quality, dtype=torch.float32)

    def _teacher_skill_preserve_indices(self, teacher_batch):
        """Rows used to preserve the with-skill teacher arm.

        This is intentionally separate from the internalization/CF active gate:
        a successful teacher trajectory should keep the with-skill behavior
        alive even when its no-skill utility is small or negative.
        """
        decisions = teacher_batch.meta_info.get("skill_int_teacher_decisions", {}) or {}
        if not decisions:
            return [], []
        cfg = self._skill_loss_cfg()
        gate = str(cfg.get("gate", "teacher_success"))
        teacher_cfg = self._teacher_rollout_cfg() or {}
        utility_eps = float(teacher_cfg.get("utility_eps", 0.0))
        aux_gate_mode = str(teacher_cfg.get("aux_gate_mode", "success"))
        fields = teacher_batch.non_tensor_batch
        group_values = np.asarray(fields.get("uid", []), dtype=object)
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        selected = []
        utilities = []
        for i, group_uid_raw in enumerate(group_values):
            group_uid = str(group_uid_raw)
            decision = decisions.get(group_uid)
            if decision is None:
                continue
            teacher_success = bool(decision.get("teacher_success", False))
            utility = float(decision.get("utility", 0.0))
            traj_uid = str(traj_values[i]) if len(traj_values) else None
            eligible_trajs = self._decision_teacher_traj_pool(decision)
            if eligible_trajs and traj_uid not in eligible_trajs:
                continue
            if gate == "teacher_success":
                keep = teacher_success
            elif gate == "active":
                keep = _teacher_aux_gate_active(aux_gate_mode, teacher_success, utility, utility_eps)
            elif gate == "utility":
                keep = utility > utility_eps
            elif gate == "success_or_utility":
                keep = teacher_success or utility > utility_eps
            elif gate == "always":
                keep = True
            else:
                raise ValueError(f"Unknown skill_sd.skill_loss.gate: {gate}")
            if not keep:
                continue
            selected.append(i)
            utilities.append(utility)
        return selected, utilities

    def _build_teacher_aux_no_skill_tensors(self, teacher_batch):
        required = ("alt_input_ids", "alt_attention_mask")
        if any(key not in teacher_batch.batch.keys() for key in required):
            return None
        responses = teacher_batch.batch["responses"]
        response_len = responses.shape[-1]
        target_len = teacher_batch.batch["input_ids"].shape[-1]
        alt_ids = teacher_batch.batch["alt_input_ids"]
        alt_mask = teacher_batch.batch["alt_attention_mask"]
        if alt_ids.shape[-1] == target_len:
            input_ids = alt_ids
            attention_mask = alt_mask
            if "alt_position_ids" in teacher_batch.batch.keys() and teacher_batch.batch["alt_position_ids"].shape[-1] == target_len:
                position_ids = teacher_batch.batch["alt_position_ids"]
            else:
                position_ids = compute_position_id_with_mask(attention_mask)
        elif alt_ids.shape[-1] + response_len == target_len:
            response_mask = teacher_batch.batch["attention_mask"][:, -response_len:]
            input_ids = torch.cat([alt_ids, responses], dim=-1)
            attention_mask = torch.cat([alt_mask, response_mask], dim=-1)
            position_ids = compute_position_id_with_mask(attention_mask)
        else:
            raise ValueError(
                "teacher aux no-skill prompt has incompatible length: "
                f"alt={alt_ids.shape[-1]}, response={response_len}, target={target_len}"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
        }

    def _slice_for_aux_shape(self, value: torch.Tensor, n_rows: int) -> torch.Tensor:
        return torch.zeros((n_rows, *value.shape[1:]), dtype=value.dtype, device=value.device)

    def _group_token_mean_weights(self, proto, response_mask: torch.Tensor) -> torch.Tensor:
        n_rows = len(proto.batch)
        if n_rows <= 0 or not bool(self.skill_sd_cfg.get("normalize_aux_by_group", True)):
            return torch.ones(n_rows, dtype=torch.float32)
        group_values = proto.non_tensor_batch.get("uid", [])
        groups = [str(self._field_at({"x": group_values}, "x", row, f"row:{row}")) for row in range(n_rows)]
        token_counts = response_mask.detach().cpu().to(torch.float32).sum(dim=-1)
        totals = {}
        for group_uid, token_count in zip(groups, token_counts.tolist()):
            totals[group_uid] = totals.get(group_uid, 0.0) + float(token_count)
        positive_groups = [group_uid for group_uid, total in totals.items() if total > 0.0]
        total_tokens = float(token_counts.sum().item())
        if total_tokens <= 0.0 or not positive_groups:
            return torch.ones(n_rows, dtype=torch.float32)
        group_count = float(len(positive_groups))
        return torch.as_tensor(
            [
                total_tokens / (group_count * totals[group_uid]) if totals[group_uid] > 0.0 else 0.0
                for group_uid in groups
            ],
            dtype=torch.float32,
        )

    def _pair_group_mean_weights(self, records) -> torch.Tensor:
        n_pairs = len(records)
        if n_pairs <= 0 or not bool(self.skill_sd_cfg.get("normalize_aux_by_group", True)):
            return torch.ones(n_pairs, dtype=torch.float32)
        groups = [str(record.get("group_uid", record.get("pair_id", idx))) for idx, record in enumerate(records)]
        counts = {}
        for group_uid in groups:
            counts[group_uid] = counts.get(group_uid, 0) + 1
        group_count = float(len(counts))
        return torch.as_tensor(
            [n_pairs / (group_count * counts[group_uid]) for group_uid in groups],
            dtype=torch.float32,
        )

    def _task_identity(self, fields, idx: int) -> Optional[str]:
        instruction = self._field_at(fields, "instruction_text", idx, None)
        if instruction is not None:
            text = str(instruction).strip()
            if text:
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                return f"instruction:{digest}"
        for key in ("task_id", "session", "extra.gamefile", "seed"):
            value = self._field_at(fields, key, idx, None)
            if value is not None:
                return f"{key}:{value}"
        uid = self._field_at(fields, "uid", idx, None)
        return None if uid is None else f"uid:{uid}"

    def _same_known_task_identity(self, teacher_fields, teacher_idx: int, student_fields, student_idx: int) -> bool:
        teacher_task = self._task_identity(teacher_fields, teacher_idx)
        student_task = self._task_identity(student_fields, student_idx)
        if teacher_task is None or student_task is None:
            return True
        return teacher_task == student_task

    def _fixed_g_cfg(self):
        if self.skill_sd_cfg is None:
            return {}
        return self.skill_sd_cfg.get("fixed_g", {}) or {}

    def _fixed_g_enabled(self) -> bool:
        return bool(self._fixed_g_cfg().get("enabled", False))

    def _fixed_g_source(self) -> str:
        return str(self._fixed_g_cfg().get("source", "val_teacher"))

    def _fixed_g_persist_enabled(self) -> bool:
        return bool(self._fixed_g_cfg().get("persist_bank", True))

    def _fixed_g_bank_path(self) -> Optional[str]:
        if not self._fixed_g_enabled() or not self._fixed_g_persist_enabled():
            return None
        explicit = self._fixed_g_cfg().get("bank_path", None)
        if explicit:
            return str(explicit)
        return self._default_skill_sd_dir("fixed_g_bank.pt")

    def _fixed_g_explicit_task_ids(self) -> Optional[list[str]]:
        explicit = self._fixed_g_cfg().get("task_ids", None)
        if not explicit:
            return None
        return [str(x) for x in explicit]

    def _fixed_g_explicit_ids(self) -> Optional[set[str]]:
        explicit = self._fixed_g_explicit_task_ids()
        if explicit is None:
            return None
        return set(explicit)

    def _fixed_g_capacity(self) -> int:
        explicit = self._fixed_g_explicit_task_ids()
        if explicit is not None:
            return max(len(explicit), 1)
        return max(int(self._fixed_g_cfg().get("sample_n", 32)), 1)

    def _fixed_g_accepts_task(self, task_id: str) -> bool:
        explicit = self._fixed_g_explicit_ids()
        if explicit is not None:
            return task_id in explicit
        return task_id in self._skill_int_fixed_ref_ids or len(self._skill_int_fixed_ref_order) < self._fixed_g_capacity()

    def _fixed_g_bank_ready(self) -> bool:
        if not self._fixed_g_enabled():
            return False
        explicit = self._fixed_g_explicit_ids()
        if explicit is not None:
            return explicit.issubset(set(self._skill_int_fixed_ref_bank.keys()))
        return len(self._skill_int_fixed_ref_bank) >= self._fixed_g_capacity()

    def _load_fixed_g_bank(self) -> None:
        path = self._fixed_g_bank_path()
        if path is None or not os.path.exists(path):
            return
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            bank = payload.get("bank", {}) if isinstance(payload, dict) else {}
            order = payload.get("order", []) if isinstance(payload, dict) else []
            ids = payload.get("ids", order) if isinstance(payload, dict) else order
            if not isinstance(bank, dict):
                return
            self._skill_int_fixed_ref_bank = bank
            self._skill_int_fixed_ref_order = [str(x) for x in order if str(x) in bank]
            self._skill_int_fixed_ref_ids = set(str(x) for x in ids) | set(self._skill_int_fixed_ref_order)
            print(
                f"[SkillSDTrainer] loaded fixed-G bank: tasks={len(self._skill_int_fixed_ref_order)} <- {path}",
                flush=True,
            )
        except Exception as err:
            print(f"[SkillSDTrainer] failed to load fixed-G bank (non-fatal): {err}", flush=True)

    def _save_fixed_g_bank(self) -> None:
        path = self._fixed_g_bank_path()
        if path is None:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "source": self._fixed_g_source(),
                "capacity": self._fixed_g_capacity(),
                "order": list(self._skill_int_fixed_ref_order),
                "ids": list(self._skill_int_fixed_ref_ids),
                "bank": self._skill_int_fixed_ref_bank,
            }
            torch.save(payload, path)
        except Exception as err:
            print(f"[SkillSDTrainer] failed to save fixed-G bank (non-fatal): {err}", flush=True)

    def _fixed_val_enabled(self) -> bool:
        cfg = self.skill_sd_cfg.get("fixed_val", {}) if self.skill_sd_cfg is not None else {}
        env_name = str(getattr(self.config, "env", {}).get("env_name", ""))
        return bool((cfg or {}).get("enabled", False)) and env_name.lower() == "webshop"

    @contextmanager
    def _fixed_webshop_val_sessions(self) -> Iterator[None]:
        if not self._fixed_val_enabled() or not hasattr(self, "traj_collector"):
            yield
            return

        cfg = self.skill_sd_cfg.get("fixed_val", {}) or {}
        session_start = int(cfg.get("webshop_session_start", 0))
        original_loop = self.traj_collector.multi_turn_loop
        val_envs = getattr(self, "val_envs", None)
        offset = 0

        def fixed_session_loop(gen_batch, actor_rollout_wg, envs, is_train=True, **kw):
            nonlocal offset
            if (not is_train) and envs is val_envs:
                n_rows = len(gen_batch.batch)
                existing = gen_batch.non_tensor_batch.get("env_kwargs", [None] * n_rows)
                env_kwargs = []
                for i in range(n_rows):
                    item = existing[i] if i < len(existing) else None
                    item = dict(item) if isinstance(item, dict) else {}
                    item["session"] = session_start + offset + i
                    item["_skill_int_explicit_sessions"] = True
                    env_kwargs.append(item)
                gen_batch.non_tensor_batch["env_kwargs"] = np.asarray(env_kwargs, dtype=object)
                offset += n_rows
            return original_loop(gen_batch, actor_rollout_wg, envs, is_train=is_train, **kw)

        self.traj_collector.multi_turn_loop = fixed_session_loop
        try:
            yield
        finally:
            self.traj_collector.multi_turn_loop = original_loop

    def _restart_alfworld_val_game_sequences(self) -> bool:
        env_name = str(getattr(self.config, "env", {}).get("env_name", "")).lower()
        if not env_name.startswith("alfworld"):
            return False
        val_envs = getattr(self, "val_envs", None)
        env_backend = getattr(val_envs, "envs", None)
        restart = getattr(env_backend, "restart_game_sequences", None)
        if not callable(restart):
            raise RuntimeError("ALFWorld validation backend cannot restart game sequences")
        restart()
        return True

    def _sync_fixed_ref_ids(self, teacher_batch) -> None:
        if not self._fixed_g_enabled():
            return
        explicit = self._fixed_g_explicit_task_ids()
        if explicit is not None:
            for task_id in explicit:
                if task_id not in self._skill_int_fixed_ref_ids:
                    self._skill_int_fixed_ref_ids.add(task_id)
                    self._skill_int_fixed_ref_order.append(task_id)
            return
        sample_n = self._fixed_g_capacity()
        fields = teacher_batch.non_tensor_batch
        for i in range(len(teacher_batch.batch)):
            if len(self._skill_int_fixed_ref_order) >= sample_n:
                break
            task_id = self._task_identity(fields, i)
            if task_id is not None and str(task_id) not in self._skill_int_fixed_ref_ids:
                self._skill_int_fixed_ref_ids.add(str(task_id))
                self._skill_int_fixed_ref_order.append(str(task_id))

    def _fixed_ref_mask_for_teacher(self, teacher_batch) -> torch.Tensor:
        n_rows = len(teacher_batch.batch)
        if not self._fixed_g_enabled():
            return torch.zeros(n_rows, dtype=torch.bool)
        self._sync_fixed_ref_ids(teacher_batch)
        fields = teacher_batch.non_tensor_batch
        mask = []
        for i in range(n_rows):
            task_id = self._task_identity(fields, i)
            mask.append(task_id is not None and str(task_id) in self._skill_int_fixed_ref_ids)
        return torch.as_tensor(mask, dtype=torch.bool)

    def _clone_dataproto_cpu(self, proto) -> DataProto:
        tensors = {
            key: value.detach().cpu().clone()
            for key, value in proto.batch.items()
        }
        non_tensors = {
            key: np.asarray(value, dtype=object).copy()
            for key, value in proto.non_tensor_batch.items()
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(proto.meta_info))

    def _register_fixed_g_refs(self, teacher_batch) -> bool:
        """Freeze a small bank of teacher demos for stable G-curve probing."""
        if not self._fixed_g_enabled() or teacher_batch is None:
            return False
        fields = teacher_batch.non_tensor_batch
        rows_by_task: Dict[str, list[int]] = {}
        order = []
        for i in range(len(teacher_batch.batch)):
            task_id = self._task_identity(fields, i)
            if task_id is None:
                continue
            task_id = str(task_id)
            if task_id in self._skill_int_fixed_ref_bank:
                continue
            if not self._fixed_g_accepts_task(task_id):
                continue
            if task_id not in rows_by_task:
                rows_by_task[task_id] = []
                order.append(task_id)
            rows_by_task[task_id].append(i)

        changed = False
        for task_id in order:
            if task_id in self._skill_int_fixed_ref_bank:
                continue
            if self._fixed_g_explicit_ids() is None and len(self._skill_int_fixed_ref_order) >= self._fixed_g_capacity():
                break
            rows = np.asarray(rows_by_task[task_id], dtype=np.int64)
            self._skill_int_fixed_ref_bank[task_id] = self._clone_dataproto_cpu(teacher_batch.select_idxs(rows))
            changed = True
            if task_id not in self._skill_int_fixed_ref_ids:
                self._skill_int_fixed_ref_ids.add(task_id)
            if task_id not in self._skill_int_fixed_ref_order:
                self._skill_int_fixed_ref_order.append(task_id)
        if changed:
            self._save_fixed_g_bank()
        return changed

    def _fixed_g_proto_signature(self, proto) -> tuple:
        batch_sig = tuple(
            (key, tuple(value.shape[1:]), str(value.dtype))
            for key, value in sorted(proto.batch.items())
        )
        non_tensor_sig = tuple(sorted(proto.non_tensor_batch.keys()))
        return batch_sig, non_tensor_sig

    def _make_no_skill_logprob_proto(self, teacher_proto):
        no_skill = self._build_teacher_aux_no_skill_tensors(teacher_proto)
        if no_skill is None:
            return None
        tensors = {
            key: value.detach().cpu().clone()
            for key, value in teacher_proto.batch.items()
        }
        for key, value in no_skill.items():
            tensors[key] = value.detach().cpu().clone()
        non_tensors = {
            key: np.asarray(value, dtype=object).copy()
            for key, value in teacher_proto.non_tensor_batch.items()
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(teacher_proto.meta_info))

    def _compute_fixed_g_probe_metrics(self) -> Dict[str, float]:
        if not self._fixed_g_enabled():
            self._last_fixed_g_record = None
            return {}
        if not self._skill_int_fixed_ref_bank:
            record = {
                "global_step": int(getattr(self, "global_steps", 0)),
                "fixed_ref_ids": list(self._skill_int_fixed_ref_order),
                "fixed_G_source": self._fixed_g_source(),
                "fixed_G_capacity": float(self._fixed_g_capacity()),
                "fixed_G_ready": float(self._fixed_g_bank_ready()),
                "fixed_G_mean": None,
                "fixed_G_std": None,
                "fixed_with_skill_logp_mean": None,
                "fixed_no_skill_logp_mean": None,
                "fixed_G_count": 0.0,
                "fixed_G_task_count": 0.0,
                "fixed_G_by_ref": [],
            }
            self._last_fixed_g_record = record
            return {
                "skill_int/fixed_ref_tasks": float(len(self._skill_int_fixed_ref_order)),
                "skill_int/fixed_G_ready": float(self._fixed_g_bank_ready()),
                "skill_int/fixed_G_capacity": float(self._fixed_g_capacity()),
                "skill_int/fixed_G_count": 0.0,
                "skill_int/fixed_G_task_count": 0.0,
            }

        try:
            groups: Dict[tuple, list] = {}
            for task_id in self._skill_int_fixed_ref_order:
                proto = self._skill_int_fixed_ref_bank.get(task_id)
                if proto is None:
                    continue
                groups.setdefault(self._fixed_g_proto_signature(proto), []).append((task_id, proto))

            all_g = []
            all_with_skill_logp = []
            all_no_skill_logp = []
            all_task_ids = []
            by_ref = []
            token_count = 0.0
            for refs in groups.values():
                task_ids = [task_id for task_id, _ in refs]
                teacher_proto = DataProto.concat([proto for _, proto in refs])
                no_skill_proto = self._make_no_skill_logprob_proto(teacher_proto)
                if no_skill_proto is None:
                    continue
                teacher_log_probs = self._compute_log_probs_for_proto(teacher_proto).detach().cpu()
                no_skill_log_probs = self._compute_log_probs_for_proto(no_skill_proto).detach().cpu()
                response_len = int(teacher_proto.batch["responses"].shape[-1])
                response_mask = no_skill_proto.batch["attention_mask"][:, -response_len:].detach().cpu()
                mask = response_mask.to(teacher_log_probs.dtype)
                denom = mask.sum(dim=-1).clamp(min=1.0)
                row_with_skill_logp = (teacher_log_probs * mask).sum(dim=-1) / denom
                row_no_skill_logp = (no_skill_log_probs * mask).sum(dim=-1) / denom
                row_g = row_with_skill_logp - row_no_skill_logp
                token_count += float(mask.sum().item())

                row_task_ids = [
                    str(self._task_identity(teacher_proto.non_tensor_batch, i) or f"row:{i}")
                    for i in range(len(teacher_proto.batch))
                ]
                row_g_values = row_g.tolist()
                row_with_values = row_with_skill_logp.tolist()
                row_no_values = row_no_skill_logp.tolist()
                all_g.extend(float(x) for x in row_g_values)
                all_with_skill_logp.extend(float(x) for x in row_with_values)
                all_no_skill_logp.extend(float(x) for x in row_no_values)
                all_task_ids.extend(row_task_ids)
                for task_id in task_ids:
                    row_triplets = [
                        (g, with_logp, no_logp)
                        for g, with_logp, no_logp, tid in zip(row_g_values, row_with_values, row_no_values, row_task_ids)
                        if tid == task_id
                    ]
                    vals = [g for g, _, _ in row_triplets]
                    if vals:
                        by_ref.append({
                            "task_id": task_id,
                            "row_count": float(len(vals)),
                            "G_mean": float(np.mean(vals)),
                            "with_skill_logp_mean": float(np.mean([x for _, x, _ in row_triplets])),
                            "no_skill_logp_mean": float(np.mean([x for _, _, x in row_triplets])),
                        })

            if not all_g:
                raise RuntimeError("fixed G bank has no valid no-skill probe rows")
            arr = np.asarray(all_g, dtype=float)
            with_arr = np.asarray(all_with_skill_logp, dtype=float)
            no_arr = np.asarray(all_no_skill_logp, dtype=float)
            task_count = float(len(set(all_task_ids)))
            record = {
                "global_step": int(getattr(self, "global_steps", 0)),
                "fixed_ref_ids": list(self._skill_int_fixed_ref_order),
                "fixed_G_source": self._fixed_g_source(),
                "fixed_G_capacity": float(self._fixed_g_capacity()),
                "fixed_G_ready": float(self._fixed_g_bank_ready()),
                "fixed_G_mean": float(arr.mean()),
                "fixed_G_std": float(arr.std()) if arr.size > 1 else 0.0,
                "fixed_with_skill_logp_mean": float(with_arr.mean()),
                "fixed_no_skill_logp_mean": float(no_arr.mean()),
                "fixed_G_count": float(arr.size),
                "fixed_G_task_count": task_count,
                "fixed_G_token_count": token_count,
                "fixed_G_by_ref": by_ref,
            }
            self._last_fixed_g_record = record
            return {
                "skill_int/fixed_ref_tasks": float(len(self._skill_int_fixed_ref_order)),
                "skill_int/fixed_G_ready": float(self._fixed_g_bank_ready()),
                "skill_int/fixed_G_capacity": float(self._fixed_g_capacity()),
                "skill_int/fixed_G_mean": float(arr.mean()),
                "skill_int/fixed_G_std": float(arr.std()) if arr.size > 1 else 0.0,
                "skill_int/fixed_with_skill_logp_mean": float(with_arr.mean()),
                "skill_int/fixed_no_skill_logp_mean": float(no_arr.mean()),
                "skill_int/fixed_G_count": float(arr.size),
                "skill_int/fixed_G_task_count": task_count,
                "skill_int/fixed_G_token_count": token_count,
            }
        except Exception as err:
            print(f"[SkillSDTrainer] fixed G probe failed (non-fatal): {err}", flush=True)
            self._last_fixed_g_record = {
                "global_step": int(getattr(self, "global_steps", 0)),
                "fixed_ref_ids": list(self._skill_int_fixed_ref_order),
                "fixed_G_source": self._fixed_g_source(),
                "fixed_G_capacity": float(self._fixed_g_capacity()),
                "fixed_G_ready": float(self._fixed_g_bank_ready()),
                "fixed_G_mean": None,
                "fixed_with_skill_logp_mean": None,
                "fixed_no_skill_logp_mean": None,
                "fixed_G_count": 0.0,
                "fixed_G_task_count": 0.0,
                "error": str(err),
            }
            return {
                "skill_int/fixed_ref_tasks": float(len(self._skill_int_fixed_ref_order)),
                "skill_int/fixed_G_ready": float(self._fixed_g_bank_ready()),
                "skill_int/fixed_G_capacity": float(self._fixed_g_capacity()),
                "skill_int/fixed_G_probe_failed": 1.0,
            }

    def _make_teacher_aux_proto(
        self,
        student_batch,
        teacher_batch,
        teacher_log_probs,
        utilities,
        quality_weights=None,
    ):
        no_skill = self._build_teacher_aux_no_skill_tensors(teacher_batch)
        if no_skill is None:
            return None
        n_rows = len(teacher_batch.batch)
        response_len = student_batch.batch["responses"].shape[-1]
        tensors = {}
        for key, value in student_batch.batch.items():
            if key in no_skill and tuple(no_skill[key].shape[1:]) == tuple(value.shape[1:]):
                tensors[key] = no_skill[key].to(dtype=value.dtype)
            else:
                tensors[key] = self._slice_for_aux_shape(value, n_rows)
        tensors["input_ids"] = no_skill["input_ids"].to(dtype=student_batch.batch["input_ids"].dtype)
        tensors["attention_mask"] = no_skill["attention_mask"].to(dtype=student_batch.batch["attention_mask"].dtype)
        tensors["position_ids"] = no_skill["position_ids"].to(dtype=student_batch.batch["position_ids"].dtype)
        tensors["responses"] = no_skill["responses"].to(dtype=student_batch.batch["responses"].dtype)
        response_mask = no_skill["attention_mask"][:, -response_len:]
        tensors["old_log_probs"] = torch.zeros((n_rows, response_len), dtype=teacher_log_probs.dtype)
        tensors["advantages"] = torch.zeros((n_rows, response_len), dtype=teacher_log_probs.dtype)
        if "response_mask" in student_batch.batch.keys():
            tensors["response_mask"] = response_mask.to(student_batch.batch["response_mask"].dtype)
        if "loss_mask" in student_batch.batch.keys():
            loss_mask = torch.zeros_like(no_skill["attention_mask"])
            loss_mask[:, -response_len:] = response_mask.to(loss_mask.dtype)
            tensors["loss_mask"] = loss_mask.to(student_batch.batch["loss_mask"].dtype)
        tensors["skill_int_is_teacher_aux"] = torch.ones(n_rows, dtype=torch.bool)
        tensors["skill_int_is_teacher_demo"] = torch.ones(n_rows, dtype=torch.bool)
        tensors["skill_int_is_skill_demo"] = torch.zeros(n_rows, dtype=torch.bool)
        tensors["skill_int_utility"] = torch.as_tensor(utilities, dtype=torch.float32)
        if quality_weights is None:
            quality_weights = self._teacher_aux_quality_weights(teacher_batch)
        quality_weights = torch.as_tensor(quality_weights, dtype=torch.float32).view(-1)
        if quality_weights.numel() != n_rows:
            raise ValueError(f"Expected {n_rows} teacher quality weights, got {quality_weights.numel()}")
        tensors["skill_int_group_weight"] = (
            self._group_token_mean_weights(teacher_batch, response_mask) * quality_weights
        )
        tensors["skill_int_teacher_log_probs"] = teacher_log_probs.float()
        tensors["skill_int_skill_old_log_probs"] = torch.zeros_like(teacher_log_probs).float()
        tensors["skill_int_cf_pair_idx"] = torch.full((n_rows,), -1, dtype=torch.long)
        tensors["skill_int_cf_role"] = torch.zeros(n_rows, dtype=torch.long)
        tensors["skill_int_fixed_ref_mask"] = self._fixed_ref_mask_for_teacher(teacher_batch)

        non_tensors = {}
        for key, values in student_batch.non_tensor_batch.items():
            if key in teacher_batch.non_tensor_batch:
                non_tensors[key] = np.asarray(teacher_batch.non_tensor_batch[key], dtype=object)
            else:
                non_tensors[key] = np.asarray([None] * n_rows, dtype=object)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=student_batch.meta_info)

    def _cf_loss_weight(self) -> float:
        if self.skill_sd_cfg is None or not bool(self.skill_sd_cfg.get("enable_cf", False)):
            return 0.0
        return float(self.skill_sd_cfg.get("lambda_cf", 0.0))

    def _skill_loss_cfg(self):
        if self.skill_sd_cfg is None:
            return {}
        return self.skill_sd_cfg.get("skill_loss", {}) or {}

    def _skill_loss_weight(self) -> float:
        if self.skill_sd_cfg is None:
            return 0.0
        return float(self.skill_sd_cfg.get("lambda_skill", 0.0))

    def _ensure_skill_int_aux_tensors(self, student_batch) -> None:
        n_rows = len(student_batch.batch)
        response_shape = student_batch.batch["old_log_probs"].shape
        defaults = {
            "skill_int_is_teacher_aux": torch.zeros(n_rows, dtype=torch.bool),
            "skill_int_is_teacher_demo": torch.zeros(n_rows, dtype=torch.bool),
            "skill_int_is_skill_demo": torch.zeros(n_rows, dtype=torch.bool),
            "skill_int_is_rescue_suffix": torch.zeros(n_rows, dtype=torch.bool),
            "skill_int_is_rescue_pg": torch.zeros(n_rows, dtype=torch.bool),
            "skill_int_utility": torch.zeros(n_rows, dtype=torch.float32),
            "skill_int_group_weight": torch.zeros(n_rows, dtype=torch.float32),
            "skill_int_teacher_log_probs": torch.zeros(response_shape, dtype=student_batch.batch["old_log_probs"].dtype).float(),
            "skill_int_skill_old_log_probs": torch.zeros(response_shape, dtype=student_batch.batch["old_log_probs"].dtype).float(),
            "skill_int_cf_pair_idx": torch.full((n_rows,), -1, dtype=torch.long),
            "skill_int_cf_role": torch.zeros(n_rows, dtype=torch.long),
            "skill_int_fixed_ref_mask": torch.zeros(n_rows, dtype=torch.bool),
            "skill_int_suffix_q_input_ids": torch.zeros_like(
                student_batch.batch["input_ids"]
            ),
            "skill_int_suffix_q_attention_mask": torch.zeros_like(
                student_batch.batch["attention_mask"]
            ),
            "skill_int_suffix_q_position_ids": torch.zeros_like(
                student_batch.batch["position_ids"]
            ),
            "skill_int_suffix_action_mask": torch.zeros_like(
                student_batch.batch["responses"]
            ),
            "skill_int_suffix_archive_node_hash": torch.zeros(
                n_rows, dtype=torch.long
            ),
            "skill_int_suffix_archive_step": torch.full(
                (n_rows,), -1, dtype=torch.long
            ),
        }
        for key, value in defaults.items():
            if key not in student_batch.batch.keys():
                student_batch.batch[key] = value

    def _cf_pair_record_value(self, pair, key, default=None):
        if isinstance(pair, dict):
            return pair.get(key, default)
        return getattr(pair, key, default)

    def _cf_row_key(self, fields, row: int):
        group_uid = self._field_at(fields, "uid", row, None)
        traj_uid = self._field_at(fields, "traj_uid", row, None)
        step = self._field_at(fields, "step_idx", row, None)
        if group_uid is None or traj_uid is None or step is None:
            return None
        step = self._int_value(step, -1)
        if step < 0:
            return None
        return str(group_uid), str(traj_uid), step

    def _cf_row_lookup(self, proto):
        lookup = {}
        for row in range(len(proto.batch)):
            key = self._cf_row_key(proto.non_tensor_batch, row)
            if key is not None:
                lookup.setdefault(key, []).append(row)
        return lookup

    def _ensure_prefix_rescue_student_row_uids(self, student_batch):
        """Attach row identities that survive DataProto reordering."""
        n_rows = len(student_batch.batch)
        existing = student_batch.non_tensor_batch.get(
            PREFIX_RESCUE_STUDENT_ROW_UID, []
        )
        row_uids = []
        seen = set()
        for row in range(n_rows):
            value = existing[row] if row < len(existing) else None
            value = "" if value is None else str(value).strip()
            if not value or value in seen:
                value = uuid.uuid4().hex
            row_uids.append(value)
            seen.add(value)
        result = np.asarray(row_uids, dtype=object)
        student_batch.non_tensor_batch[PREFIX_RESCUE_STUDENT_ROW_UID] = result
        return result

    def _prefix_rescue_student_identity(self, student_batch, row: int) -> dict:
        """Capture a stable student-row identity before fit-loop rebalancing."""
        fields = student_batch.non_tensor_batch
        row_uids = self._ensure_prefix_rescue_student_row_uids(student_batch)
        return {
            "student_row_uid": str(row_uids[row]),
            "student_group_uid": self._field_at(fields, "uid", row, None),
            "student_step": self._int_value(
                self._field_at(fields, "step_idx", row, -1), -1
            ),
            "student_task_identity": self._task_identity(fields, row),
            "student_action": cf_action_at(student_batch, row),
            "student_observation": cf_observation_at(student_batch, row),
            "student_response_hash": cf_response_hash_at(student_batch, row),
        }

    def _prefix_rescue_stable_student_key(self, record):
        group_uid = record.get(
            "student_group_uid", record.get("original_group_uid", None)
        )
        traj_uid = record.get("student_traj_uid", None)
        step = self._int_value(record.get("student_step", -1), -1)
        if group_uid is None or traj_uid is None or step < 0:
            return None
        return str(group_uid), str(traj_uid), int(step)

    def _prefix_rescue_student_identity_matches(
        self, record, student_batch, row: int
    ) -> bool:
        expected_task = record.get("student_task_identity", None)
        if expected_task is not None and self._task_identity(
            student_batch.non_tensor_batch, row
        ) != str(expected_task):
            return False
        for key, actual in (
            ("student_action", cf_action_at(student_batch, row)),
            ("student_observation", cf_observation_at(student_batch, row)),
            ("student_response_hash", cf_response_hash_at(student_batch, row)),
        ):
            expected = record.get(key, None)
            if expected is not None and str(actual) != str(expected):
                return False
        return True

    def _cf_pair_stable_key(self, pair, role: str):
        group_uid = self._cf_pair_record_value(pair, "group_uid", None)
        traj_uid = self._cf_pair_record_value(pair, f"{role}_traj_uid", None)
        step = self._int_value(self._cf_pair_record_value(pair, f"{role}_step", -1), -1)
        if group_uid is None or traj_uid is None or step < 0:
            return None
        return str(group_uid), str(traj_uid), step

    def _cf_pair_row_matches(self, pair, role: str, proto, row: int) -> bool:
        expected_task = self._cf_pair_record_value(pair, f"{role}_task_id", None)
        actual_task = self._task_identity(proto.non_tensor_batch, row)
        if expected_task is not None and actual_task != str(expected_task):
            return False

        expected_action = self._cf_pair_record_value(pair, f"{role}_action", None)
        if expected_action is not None and cf_action_at(proto, row) != str(expected_action):
            return False

        expected_observation = self._cf_pair_record_value(pair, f"{role}_observation", None)
        if expected_observation is not None and cf_observation_at(proto, row) != str(expected_observation):
            return False

        expected_response_hash = self._cf_pair_record_value(pair, f"{role}_response_hash", None)
        if expected_response_hash is not None and cf_response_hash_at(proto, row) != str(expected_response_hash):
            return False
        return True

    def _valid_cf_pair_records(self, student_batch, active_teacher, active_indices):
        records = []
        pairs = active_teacher.meta_info.get("skill_int_cf_span_pairs", []) or []
        resolution = {
            "skill_int/cf_pair_candidates": 0.0,
            "skill_int/cf_pair_resolved": 0.0,
            "skill_int/cf_pair_drop_missing_teacher": 0.0,
            "skill_int/cf_pair_drop_missing_student": 0.0,
            "skill_int/cf_pair_drop_mismatch": 0.0,
            "skill_int/cf_pair_drop_duplicate": 0.0,
        }
        if not pairs:
            pair_values = active_teacher.non_tensor_batch.get("skill_int_cf_pair_id", [])
            student_trajs = active_teacher.non_tensor_batch.get("skill_int_cf_student_traj_uid", [])
            turns = active_teacher.non_tensor_batch.get("skill_int_cf_divergence_turn", [])
            utilities = active_teacher.non_tensor_batch.get("skill_int_cf_utility", [])
            groups = active_teacher.non_tensor_batch.get("uid", [])
            teacher_trajs = active_teacher.non_tensor_batch.get("traj_uid", [])
            teacher_steps = active_teacher.non_tensor_batch.get("step_idx", [])
            student_group_values = student_batch.non_tensor_batch.get("uid", [])
            student_traj_values = student_batch.non_tensor_batch.get("traj_uid", [])
            seen = set()
            for teacher_row, pair_id in enumerate(pair_values):
                if pair_id is None:
                    continue
                resolution["skill_int/cf_pair_candidates"] += 1.0
                pair_id = str(pair_id)
                if pair_id in seen:
                    resolution["skill_int/cf_pair_drop_duplicate"] += 1.0
                    continue
                group_uid = self._field_at({"x": groups}, "x", teacher_row)
                student_traj_uid = self._field_at({"x": student_trajs}, "x", teacher_row)
                matching_student_rows = [
                    row
                    for row in range(len(student_batch.batch))
                    if group_uid is not None
                    and student_traj_uid is not None
                    and str(self._field_at({"x": student_group_values}, "x", row)) == str(group_uid)
                    and str(self._field_at({"x": student_traj_values}, "x", row)) == str(student_traj_uid)
                ]
                if len(matching_student_rows) != 1:
                    resolution["skill_int/cf_pair_drop_missing_student"] += 1.0
                    continue
                student_row = matching_student_rows[0]
                if not self._same_known_task_identity(active_teacher.non_tensor_batch, teacher_row, student_batch.non_tensor_batch, student_row):
                    resolution["skill_int/cf_pair_drop_mismatch"] += 1.0
                    continue
                seen.add(pair_id)
                records.append({
                    "pair_id": pair_id,
                    "pair_idx": len(records),
                    "group_uid": group_uid,
                    "teacher_traj_uid": self._field_at({"x": teacher_trajs}, "x", teacher_row),
                    "teacher_src_row": int(teacher_row),
                    "teacher_row": int(teacher_row),
                    "teacher_step": self._int_value(self._field_at({"x": teacher_steps}, "x", teacher_row), -1),
                    "student_row": int(student_row),
                    "student_traj_uid": student_traj_uid,
                    "student_step": -1,
                    "divergence_turn": self._int_value(self._field_at({"x": turns}, "x", teacher_row), -1),
                    "utility": float(self._field_at({"x": utilities}, "x", teacher_row, 0.0) or 0.0),
                    "teacher_task_id": self._task_identity(active_teacher.non_tensor_batch, teacher_row),
                    "student_task_id": self._task_identity(student_batch.non_tensor_batch, student_row),
                })
            resolution["skill_int/cf_pair_resolved"] = float(len(records))
            self._last_cf_pair_resolution_metrics = resolution
            return records

        teacher_lookup = self._cf_row_lookup(active_teacher)
        student_lookup = self._cf_row_lookup(student_batch)
        seen = set()
        for pair in pairs:
            resolution["skill_int/cf_pair_candidates"] += 1.0
            pair_id = self._cf_pair_record_value(pair, "pair_id")
            if pair_id is None:
                resolution["skill_int/cf_pair_drop_mismatch"] += 1.0
                continue
            pair_id = str(pair_id)
            if pair_id in seen:
                resolution["skill_int/cf_pair_drop_duplicate"] += 1.0
                continue

            teacher_key = self._cf_pair_stable_key(pair, "teacher")
            if teacher_key is not None:
                teacher_rows = teacher_lookup.get(teacher_key, [])
                teacher_row = teacher_rows[0] if teacher_rows else None
            else:
                teacher_row = None
            if teacher_row is None:
                resolution["skill_int/cf_pair_drop_missing_teacher"] += 1.0
                continue

            student_key = self._cf_pair_stable_key(pair, "student")
            if student_key is not None:
                student_rows = student_lookup.get(student_key, [])
                student_row = student_rows[0] if student_rows else None
            else:
                student_row = None
            if student_row is None:
                resolution["skill_int/cf_pair_drop_missing_student"] += 1.0
                continue
            if not self._same_known_task_identity(active_teacher.non_tensor_batch, teacher_row, student_batch.non_tensor_batch, student_row):
                resolution["skill_int/cf_pair_drop_mismatch"] += 1.0
                continue
            if not self._cf_pair_row_matches(pair, "teacher", active_teacher, teacher_row):
                resolution["skill_int/cf_pair_drop_mismatch"] += 1.0
                continue
            if not self._cf_pair_row_matches(pair, "student", student_batch, student_row):
                resolution["skill_int/cf_pair_drop_mismatch"] += 1.0
                continue
            seen.add(pair_id)
            records.append({
                "pair_id": pair_id,
                "pair_idx": len(records),
                "group_uid": self._cf_pair_record_value(pair, "group_uid"),
                "teacher_traj_uid": self._cf_pair_record_value(pair, "teacher_traj_uid"),
                "teacher_src_row": self._int_value(self._cf_pair_record_value(pair, "teacher_row_idx", -1), -1),
                "teacher_row": int(teacher_row),
                "student_row": int(student_row),
                "student_traj_uid": self._cf_pair_record_value(pair, "student_traj_uid"),
                "teacher_step": self._int_value(self._cf_pair_record_value(pair, "teacher_step", -1), -1),
                "student_step": self._int_value(self._cf_pair_record_value(pair, "student_step", -1), -1),
                "divergence_turn": self._int_value(self._cf_pair_record_value(pair, "divergence_turn", -1), -1),
                "utility": float(self._cf_pair_record_value(pair, "utility", 0.0) or 0.0),
                "group_utility": self._cf_pair_record_value(pair, "group_utility", None),
                "teacher_task_id": self._task_identity(active_teacher.non_tensor_batch, teacher_row),
                "student_task_id": self._task_identity(student_batch.non_tensor_batch, student_row),
            })
        resolution["skill_int/cf_pair_resolved"] = float(len(records))
        self._last_cf_pair_resolution_metrics = resolution
        return records

    def _set_cf_non_tensors(self, proto, records):
        proto.non_tensor_batch["skill_int_cf_pair_id"] = np.asarray([r["pair_id"] for r in records], dtype=object)
        proto.non_tensor_batch["skill_int_cf_student_row"] = np.asarray([r["student_row"] for r in records], dtype=object)
        proto.non_tensor_batch["skill_int_cf_student_traj_uid"] = np.asarray([r["student_traj_uid"] for r in records], dtype=object)
        proto.non_tensor_batch["skill_int_cf_divergence_turn"] = np.asarray([r["divergence_turn"] for r in records], dtype=object)
        proto.non_tensor_batch["skill_int_cf_utility"] = np.asarray([r["utility"] for r in records], dtype=object)
        proto.non_tensor_batch["skill_int_cf_teacher_task_id"] = np.asarray([r.get("teacher_task_id") for r in records], dtype=object)
        proto.non_tensor_batch["skill_int_cf_student_task_id"] = np.asarray([r.get("student_task_id") for r in records], dtype=object)

    def _make_cf_pair_aux_proto(self, student_batch, active_teacher, aux_proto, active_indices):
        records = self._valid_cf_pair_records(student_batch, active_teacher, active_indices)
        if not records:
            return None, records
        return self._interleave_cf_pair_proto(records, role1_proto=aux_proto, role2_proto=student_batch), records

    def _align_cf_role_keys(self, proto, reference):
        """Make ``proto``'s batch/non_tensor key sets identical to ``reference``
        so DataProto.concat succeeds. Missing tensor keys are filled with zeros
        shaped from the reference (aux rows do not use PPO ratio, so zeros for
        old_log_probs/advantages are inert); extra keys (rollout_log_probs,
        alt_*, prompts from a raw generate output) are dropped. Non-tensor keys
        are aligned the same way with object-dtype zeros/empty."""
        ref_keys = set(reference.batch.keys())
        cur_keys = set(proto.batch.keys())
        for key in cur_keys - ref_keys:
            proto.batch.pop(key)
        n = proto.batch.batch_size[0]
        for key in ref_keys - set(proto.batch.keys()):
            ref_val = reference.batch[key]
            shape = (n,) + tuple(ref_val.shape[1:])
            proto.batch[key] = torch.zeros(shape, dtype=ref_val.dtype)
        ref_nt = set(reference.non_tensor_batch.keys())
        for key in set(proto.non_tensor_batch.keys()) - ref_nt:
            proto.non_tensor_batch.pop(key)
        for key in ref_nt - set(proto.non_tensor_batch.keys()):
            proto.non_tensor_batch[key] = np.asarray(
                [reference.non_tensor_batch[key][0]] * n, dtype=object)

    def _interleave_cf_pair_proto(self, records, *, role1_proto, role2_proto):
        """Build the interleaved [t0,s0,t1,s1,...] CF-pair proto that dp_actor's
        compute_cf_span_preference_loss consumes. Shared by the legacy
        full-trajectory path (_make_cf_pair_aux_proto) and branch-pair collection. Records specify: teacher_row (into role1_proto),
        student_row (into role2_proto), pair_idx, utility, and the non-tensor
        bookkeeping keys used by _set_cf_non_tensors. role-1 = teacher (with-skill
        logp already on it), role-2 = student (no-skill policy rows).
        """
        teacher_rows = np.asarray([r["teacher_row"] for r in records], dtype=np.int64)
        student_rows = np.asarray([r["student_row"] for r in records], dtype=np.int64)
        pair_idxs = torch.as_tensor([r["pair_idx"] for r in records], dtype=torch.long)
        utilities = torch.as_tensor([r["utility"] for r in records], dtype=torch.float32)
        group_weights = self._pair_group_mean_weights(records)
        n_pairs = len(records)

        cf_teacher = role1_proto.select_idxs(teacher_rows)
        cf_teacher.batch["skill_int_is_teacher_aux"] = torch.ones(n_pairs, dtype=torch.bool)
        cf_teacher.batch["skill_int_is_teacher_demo"] = torch.zeros(n_pairs, dtype=torch.bool)
        cf_teacher.batch["skill_int_is_skill_demo"] = torch.zeros(n_pairs, dtype=torch.bool)
        cf_teacher.batch["skill_int_utility"] = utilities
        cf_teacher.batch["skill_int_group_weight"] = group_weights
        cf_teacher.batch["skill_int_cf_pair_idx"] = pair_idxs
        cf_teacher.batch["skill_int_cf_role"] = torch.ones(n_pairs, dtype=torch.long)
        self._set_cf_non_tensors(cf_teacher, records)

        cf_student = role2_proto.select_idxs(student_rows)
        cf_student.batch["skill_int_is_teacher_aux"] = torch.ones(n_pairs, dtype=torch.bool)
        cf_student.batch["skill_int_is_teacher_demo"] = torch.zeros(n_pairs, dtype=torch.bool)
        cf_student.batch["skill_int_is_skill_demo"] = torch.zeros(n_pairs, dtype=torch.bool)
        cf_student.batch["skill_int_utility"] = utilities
        cf_student.batch["skill_int_group_weight"] = group_weights
        # The rollout hook has responses before log probabilities and advantages.
        resp_zeros = torch.zeros_like(cf_student.batch["responses"], dtype=torch.float32)
        cf_student.batch["skill_int_teacher_log_probs"] = resp_zeros.clone()
        cf_student.batch["skill_int_skill_old_log_probs"] = resp_zeros.clone()
        cf_student.batch["advantages"] = resp_zeros.clone()
        cf_student.batch["old_log_probs"] = resp_zeros.clone()
        cf_student.batch["skill_int_cf_pair_idx"] = pair_idxs
        cf_student.batch["skill_int_cf_role"] = torch.full((n_pairs,), 2, dtype=torch.long)
        cf_student.batch["skill_int_fixed_ref_mask"] = torch.zeros(n_pairs, dtype=torch.bool)
        self._set_cf_non_tensors(cf_student, records)

        # Teacher and student batches need identical tensor keys for concatenation.
        self._align_cf_role_keys(cf_teacher, cf_student)

        stacked = DataProto.concat([cf_teacher, cf_student])
        order = []
        for row in range(n_pairs):
            order.extend([row, n_pairs + row])
        return stacked.select_idxs(np.asarray(order, dtype=np.int64))

    def _make_skill_imitation_aux_proto(self, student_batch, active_teacher, teacher_log_probs, utilities, cf_pair_records):
        cfg = self._skill_loss_cfg()
        scope = str(cfg.get("scope", "successful_trajectories"))
        if scope == "first_divergence":
            if not cf_pair_records:
                return None, []
            records = []
            seen_rows = set()
            for record in cf_pair_records:
                if float(record.get("utility", 0.0) or 0.0) <= 0.0:
                    continue
                teacher_row = int(record.get("teacher_row", -1))
                if teacher_row < 0 or teacher_row >= len(active_teacher.batch) or teacher_row in seen_rows:
                    continue
                seen_rows.add(teacher_row)
                records.append(record)
            if not records:
                return None, []
            rows = np.asarray([int(r["teacher_row"]) for r in records], dtype=np.int64)
            row_utilities = torch.as_tensor([float(r.get("utility", 0.0) or 0.0) for r in records], dtype=torch.float32)
        elif scope in {"best_trajectory", "successful_trajectories"}:
            candidate_rows = list(range(len(active_teacher.batch)))
            if scope == "best_trajectory":
                decisions = active_teacher.meta_info.get("skill_int_teacher_decisions", {}) or {}
                fields = active_teacher.non_tensor_batch
                candidate_rows = [
                    row
                    for row in candidate_rows
                    if (
                        (decision := decisions.get(str(self._field_at(fields, "uid", row)))) is not None
                        and decision.get("best_teacher_traj_uid") is not None
                        and str(self._field_at(fields, "traj_uid", row))
                        == str(decision.get("best_teacher_traj_uid"))
                    )
                ]
            rows = np.asarray(candidate_rows, dtype=np.int64)
            records = [
                {
                    "pair_id": None,
                    "student_row": -1,
                    "student_traj_uid": None,
                    "divergence_turn": -1,
                    "utility": float(utilities[row]) if row < len(utilities) else 0.0,
                }
                for row in candidate_rows
            ]
            row_utilities = torch.as_tensor(
                [float(utilities[row]) if row < len(utilities) else 0.0 for row in candidate_rows],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown skill_sd.skill_loss.scope: {scope}")

        n_rows = int(len(rows))
        if n_rows <= 0:
            return None, []

        selected_teacher = active_teacher.select_idxs(rows)
        selected_old_log_probs = teacher_log_probs[torch.as_tensor(rows, dtype=torch.long)].detach().cpu()
        response_len = student_batch.batch["responses"].shape[-1]
        tensors = {}
        for key, value in student_batch.batch.items():
            if key in selected_teacher.batch.keys() and tuple(selected_teacher.batch[key].shape[1:]) == tuple(value.shape[1:]):
                tensors[key] = selected_teacher.batch[key].to(dtype=value.dtype)
            else:
                tensors[key] = self._slice_for_aux_shape(value, n_rows)
        for key in ("input_ids", "attention_mask", "position_ids", "responses"):
            if key not in selected_teacher.batch.keys():
                return None, []
            tensors[key] = selected_teacher.batch[key].to(dtype=student_batch.batch[key].dtype)
        response_mask = selected_teacher.batch["attention_mask"][:, -response_len:]
        tensors["old_log_probs"] = torch.zeros((n_rows, response_len), dtype=selected_old_log_probs.dtype)
        tensors["advantages"] = torch.zeros((n_rows, response_len), dtype=selected_old_log_probs.dtype)
        if "response_mask" in student_batch.batch.keys():
            tensors["response_mask"] = response_mask.to(student_batch.batch["response_mask"].dtype)
        if "loss_mask" in student_batch.batch.keys():
            loss_mask = torch.zeros_like(selected_teacher.batch["attention_mask"])
            loss_mask[:, -response_len:] = response_mask.to(loss_mask.dtype)
            tensors["loss_mask"] = loss_mask.to(student_batch.batch["loss_mask"].dtype)
        tensors["skill_int_is_teacher_aux"] = torch.ones(n_rows, dtype=torch.bool)
        tensors["skill_int_is_teacher_demo"] = torch.zeros(n_rows, dtype=torch.bool)
        tensors["skill_int_is_skill_demo"] = torch.ones(n_rows, dtype=torch.bool)
        tensors["skill_int_utility"] = row_utilities
        tensors["skill_int_group_weight"] = self._group_token_mean_weights(selected_teacher, response_mask)
        tensors["skill_int_teacher_log_probs"] = torch.zeros_like(selected_old_log_probs).float()
        tensors["skill_int_skill_old_log_probs"] = selected_old_log_probs.float()
        tensors["skill_int_cf_pair_idx"] = torch.full((n_rows,), -1, dtype=torch.long)
        tensors["skill_int_cf_role"] = torch.zeros(n_rows, dtype=torch.long)
        tensors["skill_int_fixed_ref_mask"] = torch.zeros(n_rows, dtype=torch.bool)

        non_tensors = {}
        for key, values in student_batch.non_tensor_batch.items():
            if key in selected_teacher.non_tensor_batch:
                non_tensors[key] = np.asarray(selected_teacher.non_tensor_batch[key], dtype=object)
            else:
                non_tensors[key] = np.asarray([None] * n_rows, dtype=object)
        non_tensors["skill_int_cf_pair_id"] = np.asarray([r.get("pair_id") for r in records], dtype=object)
        non_tensors["skill_int_cf_student_row"] = np.asarray([r.get("student_row", -1) for r in records], dtype=object)
        non_tensors["skill_int_cf_student_traj_uid"] = np.asarray([r.get("student_traj_uid") for r in records], dtype=object)
        non_tensors["skill_int_cf_divergence_turn"] = np.asarray([r.get("divergence_turn", -1) for r in records], dtype=object)
        non_tensors["skill_int_cf_utility"] = np.asarray([float(r.get("utility", 0.0) or 0.0) for r in records], dtype=object)
        non_tensors["skill_int_cf_teacher_task_id"] = np.asarray([r.get("teacher_task_id") for r in records], dtype=object)
        non_tensors["skill_int_cf_student_task_id"] = np.asarray([r.get("student_task_id") for r in records], dtype=object)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=student_batch.meta_info), records

    def _span_pair_probe_dump_enabled(self) -> bool:
        if self.skill_sd_cfg is None or "dump_span_pair_probes" not in self.skill_sd_cfg:
            return False
        return bool(self.skill_sd_cfg.get("dump_span_pair_probes", False))

    def _compute_log_probs_for_proto(self, proto):
        lp_input = proto.select(
            batch_keys=["responses", "input_ids", "attention_mask", "position_ids"],
            deepcopy=True,
        )
        padded, pad_size = pad_dataproto_to_divisor(
            lp_input,
            max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1),
        )
        lp_out = self.actor_rollout_wg.compute_log_prob(padded)
        lp_out = unpad_dataproto(lp_out, pad_size)
        return lp_out.batch["old_log_probs"]

    def _compute_ref_log_probs_for_proto(self, proto):
        ref_input = proto.select(
            batch_keys=["responses", "input_ids", "attention_mask", "position_ids"],
            deepcopy=True,
        )
        padded, pad_size = pad_dataproto_to_divisor(
            ref_input,
            max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1),
        )
        if not getattr(self, "ref_in_actor", False):
            ref_out = self.ref_policy_wg.compute_ref_log_prob(padded)
        else:
            ref_out = self.actor_rollout_wg.compute_ref_log_prob(padded)
        ref_out = unpad_dataproto(ref_out, pad_size)
        return ref_out.batch["ref_log_prob"]

    def _prefix_rescue_action_mask(self, proto) -> torch.Tensor:
        """Return a strict action-token mask; unparseable rows stay inert."""
        response_len = int(proto.batch["responses"].shape[-1])
        # Recover action-token boundaries from the generated response.
        base_mask = proto.batch["attention_mask"][:, -response_len:]
        marker_pairs = self._rescue_action_marker_pairs()
        if not marker_pairs:
            return torch.zeros_like(base_mask)
        from recipe.skill_sd.teacher_aux_loss import action_span_mask

        return action_span_mask(
            responses=proto.batch["responses"],
            response_mask=base_mask,
            marker_pairs=marker_pairs,
            start_marker_ids=None,
            end_marker_ids=None,
            missing_marker_policy="zero",
        )

    def _decode_one_token(self, token_id: int):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            return tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except TypeError:
            try:
                return tokenizer.decode([int(token_id)])
            except Exception:
                return None
        except Exception:
            return None

    def _batch_decode_responses(self, responses):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return [None] * int(responses.shape[0])
        try:
            return tokenizer.batch_decode(responses, skip_special_tokens=True)
        except Exception:
            return [None] * int(responses.shape[0])

    def _action_probe_token_ids(self) -> tuple[list[int], list[int]]:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return [], []
        try:
            marker = list(tokenizer.encode("<action>", add_special_tokens=False))
        except Exception:
            marker = []
        skip = []
        for text in ("\n", " "):
            try:
                skip.extend(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                continue
        return [int(x) for x in marker], sorted({int(x) for x in skip})

    def _action_span_marker_token_ids(self) -> tuple[list[int], list[int]]:
        """Token ids for <action> and </action> for action-span masking."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return [], []
        try:
            start = list(tokenizer.encode("<action>", add_special_tokens=False))
            end = list(tokenizer.encode("</action>", add_special_tokens=False))
        except Exception:
            return [], []
        return [int(x) for x in start], [int(x) for x in end]

    def _rescue_action_marker_pairs(self) -> list[tuple[list[int], list[int]]]:
        """Decision-span markers for the active environment."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return []
        text_pairs = [("<action>", "</action>")]
        if self._prefix_env_name() == "searchqa":
            text_pairs = [("<search>", "</search>"), ("<answer>", "</answer>")]
        marker_pairs = []
        for start_text, end_text in text_pairs:
            try:
                start = [int(x) for x in tokenizer.encode(start_text, add_special_tokens=False)]
                end = [int(x) for x in tokenizer.encode(end_text, add_special_tokens=False)]
            except Exception:
                continue
            if start:
                marker_pairs.append((start, end))
        return marker_pairs

    def _set_rescue_action_only_meta(self, combined) -> None:
        """Attach action-span masking flags to a proto's meta_info."""
        if not bool(self.skill_sd_cfg.get("rescue_action_only", False)):
            combined.meta_info["skill_int_rescue_action_only"] = False
            return
        marker_pairs = self._rescue_action_marker_pairs()
        start_ids, end_ids = marker_pairs[0] if marker_pairs else ([], [])
        combined.meta_info["skill_int_rescue_action_only"] = bool(marker_pairs)
        combined.meta_info["skill_int_action_start_token_ids"] = start_ids
        combined.meta_info["skill_int_action_end_token_ids"] = end_ids
        combined.meta_info["skill_int_action_marker_pairs"] = marker_pairs

    def _dump_cf_span_pair_probes(self, cf_pair_proto, records):
        if cf_pair_proto is None or not records or not self._span_pair_probe_dump_enabled():
            return {}
        dump_dir = self._resolve_span_dump_dir()
        if dump_dir is None:
            return {}
        try:
            no_skill_log_probs = self._compute_log_probs_for_proto(cf_pair_proto).detach().cpu()
            teacher_log_probs = cf_pair_proto.batch["skill_int_teacher_log_probs"].detach().cpu()
            responses = cf_pair_proto.batch["responses"].detach().cpu()
            response_len = int(responses.shape[-1])
            if "loss_mask" in cf_pair_proto.batch.keys():
                response_mask = cf_pair_proto.batch["loss_mask"][:, -response_len:].detach().cpu()
            else:
                response_mask = cf_pair_proto.batch["attention_mask"][:, -response_len:].detach().cpu()
            pair_idx = cf_pair_proto.batch["skill_int_cf_pair_idx"].detach().cpu().long().view(-1)
            role = cf_pair_proto.batch["skill_int_cf_role"].detach().cpu().long().view(-1)
            utility = cf_pair_proto.batch["skill_int_utility"].detach().cpu().float().view(-1)
            action_marker_token_ids, action_skip_token_ids = self._action_probe_token_ids()

            mask_f = response_mask.to(no_skill_log_probs.dtype)
            denom = mask_f.sum(dim=-1).clamp(min=1.0)
            mean_no_skill = (no_skill_log_probs * mask_f).sum(dim=-1) / denom
            mean_with_skill = (teacher_log_probs * mask_f).sum(dim=-1) / denom
            span_gap = mean_with_skill - mean_no_skill
            first_idx, action_probe_found = _first_probe_token_indices(
                responses=responses,
                response_mask=response_mask,
                marker_token_ids=action_marker_token_ids,
                skip_token_ids=action_skip_token_ids,
            )
            row_arange = torch.arange(response_mask.shape[0])
            first_with_skill = teacher_log_probs[row_arange, first_idx]
            first_no_skill = no_skill_log_probs[row_arange, first_idx]
            g_first = first_with_skill - first_no_skill

            beta_cf = float(self.skill_sd_cfg.get("beta_cf", 1.0))
            g_thresh = float(self.skill_sd_cfg.get("g_thresh", 0.0))
            mode = _normalize_causal_probe_mode(self.skill_sd_cfg.get("causal_probe_mode", "off"))
            eta = float(self.skill_sd_cfg.get("causal_probe_eta", 1.0))
            response_text = self._batch_decode_responses(responses)
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"step_{self.global_steps:06d}_cf_span_probes.jsonl")

            dumped = []
            with open(path, "a", encoding="utf-8") as fh:
                for record in records:
                    pid = int(record["pair_idx"])
                    t_rows = torch.nonzero((pair_idx == pid) & (role == 1), as_tuple=False).view(-1)
                    s_rows = torch.nonzero((pair_idx == pid) & (role == 2), as_tuple=False).view(-1)
                    if t_rows.numel() == 0 or s_rows.numel() == 0:
                        continue
                    t_idx = int(t_rows[0].item())
                    s_idx = int(s_rows[0].item())
                    if mode == "one_step":
                        gamma = torch.exp(-eta * g_first[t_idx].abs()).clamp(max=1.0)
                        hard_gap_active = True
                    elif mode == "off":
                        hard_gap_active = bool((span_gap[t_idx] > g_thresh).item())
                        gamma = torch.ones((), dtype=no_skill_log_probs.dtype) if hard_gap_active else torch.zeros((), dtype=no_skill_log_probs.dtype)
                    else:
                        raise ValueError(f"Unknown causal_probe_mode: {mode}")
                    delta = mean_no_skill[t_idx] - mean_no_skill[s_idx]
                    utility_active = bool((utility[t_idx] > 0.0).item())
                    pair_active = bool(utility_active and hard_gap_active)
                    first_token_id = int(responses[t_idx, int(first_idx[t_idx].item())].item())
                    unweighted_loss = -torch.nn.functional.logsigmoid(beta_cf * delta)
                    weighted_loss = gamma * unweighted_loss if pair_active else unweighted_loss * 0.0
                    out = {
                        "global_step": int(getattr(self, "global_steps", 0)),
                        "pair_id": record.get("pair_id"),
                        "pair_idx": pid,
                        "uid": self._json_safe(record.get("group_uid")),
                        "utility": float(utility[t_idx].item()),
                        "utility_active": utility_active,
                        "causal_probe_mode": mode,
                        "causal_probe_eta": eta,
                        "action_probe_found": bool(action_probe_found[t_idx].item()),
                        "action_marker_token_ids": action_marker_token_ids,
                        "g_thresh": g_thresh,
                        "hard_gap_active": hard_gap_active,
                        "pair_active": pair_active,
                        "divergence_turn": self._json_safe(record.get("divergence_turn")),
                        "teacher": {
                            "traj_uid": self._json_safe(record.get("teacher_traj_uid")),
                            "row_idx": self._json_safe(record.get("teacher_src_row")),
                            "aux_row_idx": t_idx,
                            "step": self._json_safe(record.get("teacher_step")),
                            "task_id": self._json_safe(record.get("teacher_task_id")),
                            "response": response_text[t_idx] if 0 <= t_idx < len(response_text) else None,
                        },
                        "student": {
                            "traj_uid": self._json_safe(record.get("student_traj_uid")),
                            "row_idx": self._json_safe(record.get("student_row")),
                            "aux_row_idx": s_idx,
                            "step": self._json_safe(record.get("student_step")),
                            "task_id": self._json_safe(record.get("student_task_id")),
                            "response": response_text[s_idx] if 0 <= s_idx < len(response_text) else None,
                        },
                        "first_token": {
                            "index": int(first_idx[t_idx].item()),
                            "id": first_token_id,
                            "text": self._decode_one_token(first_token_id),
                            "logp_with_skill": float(first_with_skill[t_idx].item()),
                            "logp_no_skill": float(first_no_skill[t_idx].item()),
                        },
                        "g_first": float(g_first[t_idx].item()),
                        "gamma_d": float(gamma.item()),
                        "G_span": float(span_gap[t_idx].item()),
                        "teacher_logp_with_skill_mean": float(mean_with_skill[t_idx].item()),
                        "teacher_logp_no_skill_mean": float(mean_no_skill[t_idx].item()),
                        "student_logp_no_skill_mean": float(mean_no_skill[s_idx].item()),
                        "delta_dpo": float(delta.item()),
                        "cf_loss_unweighted": float(unweighted_loss.item()),
                        "cf_loss_weighted": float(weighted_loss.item()),
                    }
                    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                    dumped.append(out)
            if dumped:
                g_vals = np.asarray([x["g_first"] for x in dumped], dtype=float)
                gamma_vals = np.asarray([x["gamma_d"] for x in dumped], dtype=float)
                delta_vals = np.asarray([x["delta_dpo"] for x in dumped], dtype=float)
                active_vals = np.asarray([float(x["pair_active"]) for x in dumped], dtype=float)
                found_vals = np.asarray([float(x["action_probe_found"]) for x in dumped], dtype=float)
                print(f"[SkillSDTrainer] dumped {len(dumped)} CF span probes -> {path}", flush=True)
                return {
                    "skill_int/cf_probe_pairs_dumped": float(len(dumped)),
                    "skill_int/cf_probe_g_first_mean": float(g_vals.mean()),
                    "skill_int/cf_probe_g_first_abs_mean": float(np.abs(g_vals).mean()),
                    "skill_int/cf_probe_gamma_mean": float(gamma_vals.mean()),
                    "skill_int/cf_probe_delta_mean": float(delta_vals.mean()),
                    "skill_int/cf_probe_active_frac": float(active_vals.mean()),
                    "skill_int/cf_probe_action_found_frac": float(found_vals.mean()),
                }
            return {"skill_int/cf_probe_pairs_dumped": 0.0}
        except Exception as err:
            print(f"[SkillSDTrainer] CF span probe dump failed (non-fatal): {err}", flush=True)
            return {"skill_int/cf_probe_dump_failed": 1.0}

    def _make_neutral_aux_padding(self, template_proto, pad_rows):
        if pad_rows <= 0:
            return None
        pad = template_proto.select_idxs(np.zeros(pad_rows, dtype=np.int64))
        pad.batch["skill_int_is_teacher_aux"] = torch.ones(pad_rows, dtype=torch.bool)
        pad.batch["skill_int_is_teacher_demo"] = torch.zeros(pad_rows, dtype=torch.bool)
        pad.batch["skill_int_is_skill_demo"] = torch.zeros(pad_rows, dtype=torch.bool)
        pad.batch["skill_int_is_rescue_suffix"] = torch.zeros(pad_rows, dtype=torch.bool)
        pad.batch["skill_int_is_rescue_pg"] = torch.zeros(pad_rows, dtype=torch.bool)
        pad.batch["skill_int_utility"] = torch.zeros(pad_rows, dtype=torch.float32)
        pad.batch["skill_int_group_weight"] = torch.zeros(pad_rows, dtype=torch.float32)
        pad.batch["skill_int_teacher_log_probs"] = torch.zeros_like(pad.batch["skill_int_teacher_log_probs"]).float()
        pad.batch["skill_int_skill_old_log_probs"] = torch.zeros_like(pad.batch["skill_int_skill_old_log_probs"]).float()
        pad.batch["advantages"] = torch.zeros_like(pad.batch["advantages"])
        pad.batch["old_log_probs"] = torch.zeros_like(pad.batch["old_log_probs"])
        pad.batch["skill_int_cf_pair_idx"] = torch.full((pad_rows,), -1, dtype=torch.long)
        pad.batch["skill_int_cf_role"] = torch.zeros(pad_rows, dtype=torch.long)
        pad.batch["skill_int_fixed_ref_mask"] = torch.zeros(pad_rows, dtype=torch.bool)
        pad.non_tensor_batch["skill_int_cf_pair_id"] = np.asarray([None] * pad_rows, dtype=object)
        pad.non_tensor_batch["skill_int_cf_student_row"] = np.asarray([-1] * pad_rows, dtype=object)
        pad.non_tensor_batch["skill_int_cf_student_traj_uid"] = np.asarray([None] * pad_rows, dtype=object)
        pad.non_tensor_batch["skill_int_cf_divergence_turn"] = np.asarray([-1] * pad_rows, dtype=object)
        pad.non_tensor_batch["skill_int_cf_utility"] = np.asarray([0.0] * pad_rows, dtype=object)
        pad.non_tensor_batch["skill_int_cf_teacher_task_id"] = np.asarray([None] * pad_rows, dtype=object)
        pad.non_tensor_batch["skill_int_cf_student_task_id"] = np.asarray([None] * pad_rows, dtype=object)
        return pad

    def _build_worker_aux_chunks(self, aux_proto, cf_pair_proto, skill_proto, world_size):
        template_proto = aux_proto if aux_proto is not None else skill_proto
        if template_proto is None:
            if cf_pair_proto is None:
                raise ValueError("Cannot build aux chunks without any aux proto")
            template_proto = cf_pair_proto
        empty = template_proto.select_idxs(np.asarray([], dtype=np.int64))
        pieces = [[] for _ in range(world_size)]
        if cf_pair_proto is not None:
            n_pairs = len(cf_pair_proto.batch) // 2
            for pair_idx in range(n_pairs):
                worker = pair_idx % world_size
                rows = np.asarray([2 * pair_idx, 2 * pair_idx + 1], dtype=np.int64)
                pieces[worker].append(cf_pair_proto.select_idxs(rows))
        if skill_proto is not None:
            for row in range(len(skill_proto.batch)):
                worker = row % world_size
                pieces[worker].append(skill_proto.select_idxs(np.asarray([row], dtype=np.int64)))
        if aux_proto is not None:
            for row in range(len(aux_proto.batch)):
                worker = row % world_size
                pieces[worker].append(aux_proto.select_idxs(np.asarray([row], dtype=np.int64)))
        chunks = [DataProto.concat(worker_pieces) if worker_pieces else empty for worker_pieces in pieces]
        max_aux_rows = max(len(chunk.batch) for chunk in chunks)
        padded = []
        pad_aux = 0
        for chunk in chunks:
            pad_rows = max_aux_rows - len(chunk.batch)
            pad_aux += pad_rows
            if pad_rows:
                chunk = DataProto.concat([chunk, self._make_neutral_aux_padding(template_proto, pad_rows)])
            padded.append(chunk)
        return padded, pad_aux

    def _build_prefix_rescue_worker_chunks(
        self, teacher_pg, suffix_proto, world_size
    ):
        """Shard rescue PG once, but replicate aligned suffix KL on every rank."""
        world_size = max(int(world_size), 1)
        template = teacher_pg if teacher_pg is not None else suffix_proto
        if template is None:
            raise ValueError("prefix rescue worker chunks require training rows")
        empty = template.select_idxs(np.asarray([], dtype=np.int64))
        pg_pieces = [[] for _ in range(world_size)]
        if teacher_pg is not None:
            for row in range(len(teacher_pg.batch)):
                pg_pieces[row % world_size].append(
                    teacher_pg.select_idxs(np.asarray([row], dtype=np.int64))
                )
        max_pg_rows = max((len(rows) for rows in pg_pieces), default=0)
        chunks = []
        pad_rows_total = 0
        for worker in range(world_size):
            pieces = list(pg_pieces[worker])
            pad_rows = max_pg_rows - len(pg_pieces[worker])
            if pad_rows:
                pieces.append(self._make_neutral_aux_padding(template, pad_rows))
                pad_rows_total += pad_rows
            # Keep the replicated suffix after a fixed-width PG slot on every
            # rank. Otherwise ranks without a PG row see suffix microbatches at
            # different offsets and enter a different number of FSDP forwards.
            if suffix_proto is not None and len(suffix_proto.batch) > 0:
                pieces.append(suffix_proto)
            chunks.append(DataProto.concat(pieces) if pieces else empty)

        suffix_layouts = [
            tuple(chunk.batch["skill_int_is_rescue_suffix"].bool().tolist())
            for chunk in chunks
        ]
        if any(layout != suffix_layouts[0] for layout in suffix_layouts[1:]):
            raise RuntimeError(
                "prefix rescue suffix layout differs across actor ranks; "
                "distributed suffix-KL would execute mismatched collectives"
            )
        return chunks, pad_rows_total

    def _attach_prefix_rescue_training(self, student_batch, pair_proto):
        """Turn a verified prefix rescue into main PG plus suffix KL."""
        records = list(pair_proto.meta_info.get("prefix_rescue_pair_records", []) or [])
        if not records:
            return student_batch, {"prefix_rescue/augmented_groups": 0.0}

        strict_consistency = bool(
            self._prefix_rescue_cfg().get("strict_consistency", True)
        )
        expected_matched = int(
            pair_proto.meta_info.get(
                "prefix_rescue_matched_student_records", len(records)
            )
        )
        archived_pairs = int(
            pair_proto.meta_info.get(
                "prefix_rescue_archived_pair_records", len(records)
            )
        )
        if strict_consistency and not (
            expected_matched == len(records) == archived_pairs
        ):
            raise RuntimeError(
                "prefix rescue pair metadata mismatch before actor update: "
                f"records={len(records)}, matched={expected_matched}, "
                f"archived={archived_pairs}"
            )

        self._ensure_skill_int_aux_tensors(student_batch)
        for key, default in (
            ("skill_int_cf_pair_id", None), ("skill_int_cf_student_row", -1),
            ("skill_int_cf_student_traj_uid", None), ("skill_int_cf_divergence_turn", -1),
            ("skill_int_cf_utility", 0.0), ("skill_int_cf_teacher_task_id", None),
            ("skill_int_cf_student_task_id", None),
        ):
            if key not in student_batch.non_tensor_batch:
                student_batch.non_tensor_batch[key] = np.asarray(
                    [default] * len(student_batch.batch), dtype=object
                )

        student_action_mask = self._prefix_rescue_action_mask(student_batch).to(
            student_batch.batch["advantages"].dtype
        )
        student_lookup = self._cf_row_lookup(student_batch)
        student_row_uid_lookup = {}
        student_row_uids = student_batch.non_tensor_batch.get(
            PREFIX_RESCUE_STUDENT_ROW_UID, []
        )
        if len(student_row_uids) == len(student_batch.batch):
            for row, row_uid in enumerate(student_row_uids):
                if row_uid is not None and str(row_uid).strip():
                    student_row_uid_lookup.setdefault(str(row_uid), []).append(row)
        valid_by_group = {}
        seen = set()
        guard_counts = {
            "not_verified_success": 0,
            "missing_group_uid": 0,
            "duplicate_student": 0,
            "student_row_uid_not_found": 0,
            "student_row_uid_ambiguous": 0,
            "student_row_uid_identity_mismatch": 0,
            "student_stable_key_not_found": 0,
            "student_stable_key_ambiguous": 0,
            "student_identity_mismatch": 0,
            "student_row_out_of_range": 0,
            "student_action_mask_empty": 0,
        }
        row_uid_resolved = 0
        row_uid_duplicate_sets_resolved = 0
        row_uid_duplicate_copies_resolved = 0
        stable_key_resolved = 0
        stable_key_fallback = 0
        student_rows_remapped = 0
        guard_failures = []
        fatal_guard_rejections = 0
        for record_idx, record in enumerate(records):
            group_uid = str(record.get("group_uid", ""))
            archived_student_row = int(record.get("student_row", -1))
            student_row = archived_student_row
            stable_key = self._prefix_rescue_stable_student_key(record)
            student_row_uid = record.get("student_row_uid", None)
            student_row_uid = (
                str(student_row_uid).strip()
                if student_row_uid is not None else ""
            )
            reason = None
            if student_row_uid:
                matching_rows = student_row_uid_lookup.get(student_row_uid, [])
                if not matching_rows:
                    reason = "student_row_uid_not_found"
                else:
                    # adjust_batch may replicate an actor row while preserving
                    # its UID. Accept only exact copies of the archived
                    # identity; a UID spanning different identities remains a
                    # fatal consistency error.
                    identity_rows = []
                    for matching_row in matching_rows:
                        candidate_row = int(matching_row)
                        candidate_key = self._cf_row_key(
                            student_batch.non_tensor_batch, candidate_row
                        )
                        if stable_key is not None and candidate_key != stable_key:
                            continue
                        if not self._prefix_rescue_student_identity_matches(
                            record, student_batch, candidate_row
                        ):
                            continue
                        identity_rows.append(candidate_row)
                    if len(identity_rows) != len(matching_rows):
                        reason = "student_row_uid_identity_mismatch"
                    else:
                        student_row = (
                            archived_student_row
                            if archived_student_row in identity_rows
                            else min(identity_rows)
                        )
                        row_uid_resolved += 1
                        if len(identity_rows) > 1:
                            row_uid_duplicate_sets_resolved += 1
                            row_uid_duplicate_copies_resolved += (
                                len(identity_rows) - 1
                            )
                        student_rows_remapped += int(
                            student_row != archived_student_row
                        )
            elif stable_key is None:
                stable_key_fallback += 1
            else:
                matching_rows = student_lookup.get(stable_key, [])
                if not matching_rows:
                    reason = "student_stable_key_not_found"
                else:
                    # Invalid-action retries can produce more than one actor
                    # row for the same trajectory and environment step. Use
                    # the archived action/state/response identity to select
                    # the exact rollout row before declaring the key ambiguous.
                    identity_rows = [
                        int(row) for row in matching_rows
                        if self._prefix_rescue_student_identity_matches(
                            record, student_batch, int(row)
                        )
                    ]
                    if len(identity_rows) == 1:
                        student_row = identity_rows[0]
                        stable_key_resolved += 1
                        student_rows_remapped += int(
                            student_row != archived_student_row
                        )
                    elif not identity_rows:
                        reason = "student_identity_mismatch"
                    else:
                        reason = "student_stable_key_ambiguous"
            key = (
                group_uid,
                ("row_uid", student_row_uid)
                if student_row_uid
                else stable_key
                if stable_key is not None
                else (
                    str(record.get("student_traj_uid", "")),
                    archived_student_row,
                ),
            )
            verified_success = bool(record.get("teacher_full_success", False))
            if not verified_success:
                reason = "not_verified_success"
            elif not group_uid:
                reason = "missing_group_uid"
            elif key in seen:
                reason = "duplicate_student"
            elif reason is not None:
                pass
            elif student_row < 0 or student_row >= len(student_batch.batch):
                reason = "student_row_out_of_range"
            elif not self._prefix_rescue_student_identity_matches(
                record, student_batch, student_row
            ):
                reason = "student_identity_mismatch"
            elif not bool(student_action_mask[student_row].bool().any().item()):
                reason = "student_action_mask_empty"
            if reason is not None:
                guard_counts[reason] += 1
                fatal_guard_rejections += int(verified_success)
                guard_failures.append({
                    "reason": reason,
                    "pair_id": record.get("pair_id"),
                    "student_traj_uid": record.get("student_traj_uid"),
                    "student_step": record.get("student_step"),
                    "student_row_uid": student_row_uid or None,
                    "archived_student_row": archived_student_row,
                    "resolved_student_row": student_row,
                })
                continue
            seen.add(key)
            valid_by_group.setdefault(group_uid, []).append((record_idx, student_row))
        guard_metrics = {
            f"prefix_rescue/guard_{name}": float(count)
            for name, count in guard_counts.items()
        }
        guard_metrics["prefix_rescue/guard_rejected_total"] = float(
            sum(guard_counts.values())
        )
        guard_metrics["prefix_rescue/row_uid_resolved"] = float(
            row_uid_resolved
        )
        guard_metrics["prefix_rescue/row_uid_duplicate_sets_resolved"] = float(
            row_uid_duplicate_sets_resolved
        )
        guard_metrics["prefix_rescue/row_uid_duplicate_copies_resolved"] = float(
            row_uid_duplicate_copies_resolved
        )
        guard_metrics["prefix_rescue/stable_key_resolved"] = float(
            stable_key_resolved
        )
        guard_metrics["prefix_rescue/stable_key_fallback"] = float(
            stable_key_fallback
        )
        guard_metrics["prefix_rescue/student_rows_remapped"] = float(
            student_rows_remapped
        )
        if strict_consistency and fatal_guard_rejections:
            print(
                "prefix rescue pair guard dropped verified-success rows before "
                f"actor update: {guard_counts}; failures={guard_failures}"
            )
        if not valid_by_group:
            return student_batch, {
                "prefix_rescue/augmented_groups": 0.0,
                **guard_metrics,
            }

        scale = float(self._prefix_rescue_cfg().get("advantage_scale", 1.0))
        positive_only = bool(self._prefix_rescue_cfg().get("positive_pg_only", False))
        negative_rows = 0
        negative_tokens = 0
        original_student_response_masks = {}
        original_student_loss_masks = {}
        for group_uid, members in valid_by_group.items():
            if positive_only:
                break
            negative_advantage = -scale / float(np.sqrt(len(members)))
            for _, student_row in members:
                if "response_mask" in student_batch.batch.keys():
                    original_student_response_masks[student_row] = (
                        student_batch.batch["response_mask"][student_row].clone()
                    )
                if "loss_mask" in student_batch.batch.keys():
                    original_student_loss_masks[student_row] = (
                        student_batch.batch["loss_mask"][student_row].clone()
                    )
                student_batch.batch["advantages"][student_row].zero_()
                student_batch.batch["advantages"][student_row] += (
                    negative_advantage * student_action_mask[student_row]
                )
                negative_rows += 1
                negative_tokens += int(student_action_mask[student_row].sum().item())
                student_batch.batch["skill_int_is_rescue_pg"][student_row] = True
                if "response_mask" in student_batch.batch.keys():
                    student_batch.batch["response_mask"][student_row] = (
                        student_action_mask[student_row].to(
                            student_batch.batch["response_mask"].dtype
                        )
                    )
                if "loss_mask" in student_batch.batch.keys():
                    response_len = int(student_batch.batch["responses"].shape[-1])
                    student_batch.batch["loss_mask"][student_row].zero_()
                    student_batch.batch["loss_mask"][student_row, -response_len:] = (
                        student_action_mask[student_row].to(
                            student_batch.batch["loss_mask"].dtype
                        )
                    )

        teacher_pair_rows = []
        teacher_groups = []
        for group_uid, members in valid_by_group.items():
            teacher_pair_rows.append(2 * int(members[0][0]))
            teacher_groups.append(group_uid)
        teacher_pg = pair_proto.select_idxs(np.asarray(teacher_pair_rows, dtype=np.int64))
        teacher_action_mask = self._prefix_rescue_action_mask(teacher_pg).to(
            student_batch.batch["advantages"].dtype
        )
        keep = [
            row for row in range(len(teacher_pg.batch))
            if bool(teacher_action_mask[row].bool().any().item())
        ]
        if len(keep) != len(teacher_groups):
            # Roll back groups whose teacher action could not be token-masked.
            rejected = set(teacher_groups) - {teacher_groups[row] for row in keep}
            for group_uid in rejected:
                for _, student_row in valid_by_group[group_uid]:
                    student_batch.batch["advantages"][student_row].zero_()
                    student_batch.batch["skill_int_is_rescue_pg"][student_row] = False
                    if student_row in original_student_response_masks:
                        student_batch.batch["response_mask"][student_row] = (
                            original_student_response_masks[student_row]
                        )
                    if student_row in original_student_loss_masks:
                        student_batch.batch["loss_mask"][student_row] = (
                            original_student_loss_masks[student_row]
                        )
            valid_by_group = {
                group_uid: members for group_uid, members in valid_by_group.items()
                if group_uid not in rejected
            }
            teacher_groups = [teacher_groups[row] for row in keep]
            teacher_pg = teacher_pg.select_idxs(np.asarray(keep, dtype=np.int64))
            teacher_action_mask = teacher_action_mask[torch.as_tensor(keep, dtype=torch.long)]
            guard_metrics["prefix_rescue/guard_teacher_action_mask_empty"] = float(
                len(rejected)
            )
            if strict_consistency:
                raise RuntimeError(
                    "prefix rescue teacher action mask missing before actor update: "
                    f"groups={sorted(rejected)}"
                )
        if not teacher_groups:
            return student_batch, {
                "prefix_rescue/augmented_groups": 0.0,
                **guard_metrics,
            }

        if strict_consistency and (
            (not positive_only and negative_rows != expected_matched)
            or len(teacher_groups) != len(valid_by_group)
        ):
            raise RuntimeError(
                "prefix rescue PG cardinality mismatch before actor update: "
                f"matched={expected_matched}, negative_rows={negative_rows}, "
                f"groups={len(valid_by_group)}, teacher_rows={len(teacher_groups)}"
            )

        teacher_pg.batch["advantages"] = torch.stack([
            scale * float(np.sqrt(len(valid_by_group[group_uid]))) * teacher_action_mask[row]
            for row, group_uid in enumerate(teacher_groups)
        ])
        teacher_pg.batch["old_log_probs"] = self._compute_log_probs_for_proto(
            teacher_pg
        ).to(student_batch.batch["old_log_probs"].dtype)
        if "ref_log_prob" in student_batch.batch.keys():
            teacher_pg.batch["ref_log_prob"] = self._compute_ref_log_probs_for_proto(
                teacher_pg
            ).to(student_batch.batch["ref_log_prob"].dtype)
        if "response_mask" in teacher_pg.batch.keys():
            teacher_pg.batch["response_mask"] = teacher_action_mask.to(
                teacher_pg.batch["response_mask"].dtype
            )
        if "loss_mask" in teacher_pg.batch.keys():
            response_len = int(teacher_pg.batch["responses"].shape[-1])
            loss_mask = torch.zeros_like(teacher_pg.batch["loss_mask"])
            loss_mask[:, -response_len:] = teacher_action_mask.to(loss_mask.dtype)
            teacher_pg.batch["loss_mask"] = loss_mask
        n_teacher = len(teacher_pg.batch)
        teacher_pg.batch["skill_int_is_teacher_aux"] = torch.zeros(n_teacher, dtype=torch.bool)
        teacher_pg.batch["skill_int_is_teacher_demo"] = torch.zeros(n_teacher, dtype=torch.bool)
        teacher_pg.batch["skill_int_is_skill_demo"] = torch.zeros(n_teacher, dtype=torch.bool)
        teacher_pg.batch["skill_int_is_rescue_suffix"] = torch.zeros(n_teacher, dtype=torch.bool)
        teacher_pg.batch["skill_int_is_rescue_pg"] = torch.ones(n_teacher, dtype=torch.bool)
        training_signal_records = 0
        for teacher_row, group_uid in enumerate(teacher_groups):
            members = valid_by_group[group_uid]
            self._append_artifact("rescue_training_signal", {
                "local_group_uid": group_uid,
                "teacher_advantages": teacher_pg.batch["advantages"][teacher_row],
                "teacher_response_mask": teacher_action_mask[teacher_row],
                "teacher_loss_mask": (
                    teacher_pg.batch["loss_mask"][teacher_row]
                    if "loss_mask" in teacher_pg.batch.keys() else None
                ),
                "students": [
                    {
                        "student_row": int(student_row),
                        "advantages": student_batch.batch["advantages"][student_row],
                        "response_mask": student_action_mask[student_row],
                        "loss_mask": (
                            student_batch.batch["loss_mask"][student_row]
                            if "loss_mask" in student_batch.batch.keys() else None
                        ),
                    }
                    for _, student_row in members
                ],
            })
            training_signal_records += 1
        if strict_consistency and training_signal_records != len(valid_by_group):
            raise RuntimeError(
                "prefix rescue training-signal archive mismatch before actor update: "
                f"groups={len(valid_by_group)}, archived={training_signal_records}"
            )

        suffix_proto = getattr(self, "_last_skill_int_prefix_rescue_suffix_proto", None)
        suffix_rows = 0
        suffix_tokens = 0
        if suffix_proto is not None and len(suffix_proto.batch) > 0:
            suffix_groups = np.asarray(suffix_proto.non_tensor_batch.get("uid", []), dtype=object)
            keep_suffix = np.asarray([
                row for row, group_uid in enumerate(suffix_groups)
                if str(group_uid) in valid_by_group
            ], dtype=np.int64)
            suffix_proto = suffix_proto.select_idxs(keep_suffix) if keep_suffix.size else None
        if suffix_proto is not None and len(suffix_proto.batch) > 0:
            suffix_mask = self._prefix_rescue_action_mask(suffix_proto)
            active = torch.nonzero(suffix_mask.bool().any(dim=-1), as_tuple=False).view(-1)
            suffix_proto = (
                suffix_proto.select_idxs(active.detach().cpu().numpy().astype(np.int64))
                if active.numel() else None
            )
            if suffix_proto is not None:
                suffix_mask = suffix_mask[active]
                suffix_rows = len(suffix_proto.batch)
                suffix_tokens = int(suffix_mask.sum().item())
                suffix_proto.batch["skill_int_is_teacher_aux"] = torch.ones(suffix_rows, dtype=torch.bool)
                suffix_proto.batch["skill_int_is_teacher_demo"] = torch.zeros(suffix_rows, dtype=torch.bool)
                suffix_proto.batch["skill_int_is_skill_demo"] = torch.zeros(suffix_rows, dtype=torch.bool)
                suffix_proto.batch["skill_int_is_rescue_suffix"] = torch.ones(suffix_rows, dtype=torch.bool)
                suffix_proto.batch["skill_int_is_rescue_pg"] = torch.zeros(suffix_rows, dtype=torch.bool)
                suffix_proto.batch["skill_int_suffix_action_mask"] = suffix_mask.to(
                    suffix_proto.batch["responses"].dtype
                )
                suffix_proto.batch["skill_int_group_weight"] = self._group_token_mean_weights(
                    suffix_proto, suffix_mask
                )
                if "response_mask" in suffix_proto.batch.keys():
                    suffix_proto.batch["response_mask"] = suffix_mask.to(
                        suffix_proto.batch["response_mask"].dtype
                    )
                if "loss_mask" in suffix_proto.batch.keys():
                    response_len = int(suffix_proto.batch["responses"].shape[-1])
                    loss_mask = torch.zeros_like(suffix_proto.batch["loss_mask"])
                    loss_mask[:, -response_len:] = suffix_mask.to(loss_mask.dtype)
                    suffix_proto.batch["loss_mask"] = loss_mask

        self._align_cf_role_keys(teacher_pg, student_batch)
        if suffix_proto is not None and len(suffix_proto.batch) > 0:
            self._align_cf_role_keys(suffix_proto, student_batch)
        world_size = max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1)
        student_chunks = student_batch.chunk(world_size)
        extra_chunks, pad_rows = self._build_prefix_rescue_worker_chunks(
            teacher_pg, suffix_proto, world_size
        )
        combined = DataProto.concat([
            DataProto.concat([student_chunk, extra_chunk])
            for student_chunk, extra_chunk in zip(student_chunks, extra_chunks)
        ])
        combined.meta_info["global_token_num"] = torch.sum(
            combined.batch["attention_mask"], dim=-1
        ).tolist()
        combined.meta_info["skill_int_lambda_int"] = 0.0
        combined.meta_info["skill_int_lambda_cf"] = 0.0
        combined.meta_info["skill_int_lambda_skill"] = 0.0
        combined.meta_info["skill_int_rescue_suffix_kl_weight"] = float(
            self._prefix_rescue_cfg().get("suffix_kl_weight", 0.05)
        )
        combined.meta_info["skill_int_rescue_suffix_kl_top_k"] = int(
            self._prefix_rescue_cfg().get("suffix_kl_top_k", 64)
        )
        combined.meta_info["skill_int_rescue_suffix_kl_temperature"] = float(
            self._prefix_rescue_cfg().get("suffix_kl_temperature", 1.0)
        )
        combined.meta_info["skill_int_rescue_suffix_kl_eps"] = float(
            self._prefix_rescue_cfg().get("suffix_kl_eps", 1.0e-8)
        )
        artifact_dir = self._default_skill_sd_dir("artifacts")
        if artifact_dir is not None:
            if not hasattr(self, "_artifact_run_id"):
                self._artifact_run_id = uuid.uuid4().hex
                self._artifact_manifest_registered = False
            combined.meta_info["skill_int_suffix_q_archive_dir"] = artifact_dir
            combined.meta_info["skill_int_suffix_q_archive_run_id"] = str(
                self._artifact_run_id
            )
            combined.meta_info["skill_int_suffix_q_archive_step"] = int(
                getattr(self, "global_steps", 0)
            )
            self._append_artifact("q_cache_shard_manifest", {
                "path_pattern": (
                    f"q_cache_{self._artifact_run_id}_step"
                    f"{int(getattr(self, 'global_steps', 0)):06d}_rank*.jsonl.gz"
                ),
                "world_size": world_size,
                "top_k": int(
                    self._prefix_rescue_cfg().get("suffix_kl_top_k", 64)
                ),
            })
        self._set_rescue_action_only_meta(combined)
        return combined, {
            "prefix_rescue/augmented_groups": float(len(valid_by_group)),
            "prefix_rescue/augmented_teacher_rows": float(n_teacher),
            "prefix_rescue/augmented_student_rows": float(negative_rows),
            "prefix_rescue/augmented_student_tokens": float(negative_tokens),
            "prefix_rescue/suffix_rows_attached": float(suffix_rows),
            "prefix_rescue/suffix_worker_row_copies": float(suffix_rows * world_size),
            "prefix_rescue/suffix_tokens_attached": float(suffix_tokens),
            "prefix_rescue/aux_rows_padded": float(pad_rows),
            "prefix_rescue/training_signal_archive_records": float(
                training_signal_records
            ),
            **guard_metrics,
        }

    def _attach_cf_branch_only(self, student_batch, cf_pair_proto):
        """Attach counterfactual branch pairs when the whole-traj teacher arm is
        off. Only the CF loss channel is used (no L_int/L_skill). Mirrors the CF
        meta_info of _attach_teacher_aux_batch."""
        self._ensure_skill_int_aux_tensors(student_batch)
        for key, default in (
            ("skill_int_cf_pair_id", None), ("skill_int_cf_student_row", -1),
            ("skill_int_cf_student_traj_uid", None), ("skill_int_cf_divergence_turn", -1),
            ("skill_int_cf_utility", 0.0), ("skill_int_cf_teacher_task_id", None),
            ("skill_int_cf_student_task_id", None),
        ):
            if key not in student_batch.non_tensor_batch:
                student_batch.non_tensor_batch[key] = np.asarray([default] * len(student_batch.batch), dtype=object)

        # cf_pair_proto was built inside the rollout hook, BEFORE compute_log_prob
        # ran, so it lacks old_log_probs / advantages / response_mask / loss_mask
        # (it still carries rollout_log_probs / alt_* / prompts from the raw
        # generate output). student_batch here (fit loop, post compute_log_prob)
        # has the full training key set. Align the aux proto to it so the per-worker
        # concat below matches keys.
        self._align_cf_role_keys(cf_pair_proto, student_batch)

        world_size = max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1)
        student_chunks = student_batch.chunk(world_size)
        aux_chunks, pad_aux = self._build_worker_aux_chunks(None, cf_pair_proto, None, world_size)
        combined = DataProto.concat([
            DataProto.concat([sc, ac]) for sc, ac in zip(student_chunks, aux_chunks)
        ])
        combined.meta_info["global_token_num"] = torch.sum(combined.batch["attention_mask"], dim=-1).tolist()
        lambda_cf = self._cf_loss_weight()
        combined.meta_info["skill_int_lambda_int"] = 0.0
        combined.meta_info["skill_int_lambda_skill"] = 0.0
        combined.meta_info["skill_int_lambda_cf"] = float(lambda_cf)
        combined.meta_info["skill_int_beta_cf"] = float(self.skill_sd_cfg.get("beta_cf", 1.0))
        combined.meta_info["skill_int_g_thresh"] = float(self.skill_sd_cfg.get("g_thresh", 0.0))
        combined.meta_info["skill_int_cf_preference_mode"] = str(self.skill_sd_cfg.get("cf_pair_mode", "signed"))
        combined.meta_info["skill_int_causal_probe_mode"] = _normalize_causal_probe_mode(
            self.skill_sd_cfg.get("causal_probe_mode", "off"))
        combined.meta_info["skill_int_causal_probe_eta"] = float(self.skill_sd_cfg.get("causal_probe_eta", 1.0))
        action_marker_token_ids, action_skip_token_ids = self._action_probe_token_ids()
        combined.meta_info["skill_int_action_marker_token_ids"] = action_marker_token_ids
        combined.meta_info["skill_int_action_skip_token_ids"] = action_skip_token_ids
        self._set_rescue_action_only_meta(combined)
        entropy_gate_cfg = self.skill_sd_cfg.get("entropy_gate", {}) or {}
        combined.meta_info["skill_int_entropy_gate_enabled"] = bool(entropy_gate_cfg.get("enabled", False))
        combined.meta_info["skill_int_entropy_gate_threshold"] = entropy_gate_cfg.get("threshold", None)
        combined.meta_info["skill_int_entropy_gate_quantile"] = float(entropy_gate_cfg.get("quantile", 0.75))
        combined.meta_info["skill_int_entropy_gate_beta"] = float(entropy_gate_cfg.get("beta", 1.0))
        combined.meta_info["skill_int_entropy_gate_min"] = float(entropy_gate_cfg.get("min_gate", 0.0))
        n_pairs = len(cf_pair_proto.batch) // 2
        source = str(cf_pair_proto.meta_info.get("skill_int_pair_source", "cf_branch"))
        metrics = {
            "skill_int/cf_branch_attached_pairs": float(n_pairs),
            "skill_int/aux_rows": float(len(cf_pair_proto.batch)),
            "skill_int/aux_rows_padded": float(pad_aux),
        }
        if source == "prefix_rescue":
            metrics["skill_int/prefix_rescue_attached_pairs"] = float(n_pairs)
        return combined, metrics

    def _attach_teacher_aux_batch(self, student_batch):
        teacher_batch = getattr(self, "_last_skill_int_teacher_batch", None)
        cf_branch_proto = getattr(self, "_last_skill_int_cf_branch_proto", None)
        prefix_rescue_proto = getattr(self, "_last_skill_int_prefix_rescue_proto", None)
        pending_pair_proto = prefix_rescue_proto if prefix_rescue_proto is not None else cf_branch_proto
        if teacher_batch is None:
            if prefix_rescue_proto is not None:
                return self._attach_prefix_rescue_training(
                    student_batch, prefix_rescue_proto
                )
            # Attach available branch pairs to the counterfactual loss.
            if pending_pair_proto is not None and self._cf_loss_weight() > 0:
                return self._attach_cf_branch_only(student_batch, pending_pair_proto)
            fixed_metrics = self._compute_fixed_g_probe_metrics()
            return student_batch, fixed_metrics
        active_indices, utilities = self._teacher_aux_active_indices(teacher_batch)
        skill_indices, skill_utilities = self._teacher_skill_preserve_indices(teacher_batch)
        bank_indices = sorted({int(i) for i in list(active_indices) + list(skill_indices)})
        if bank_indices and self._fixed_g_source() == "train_teacher":
            teacher_for_bank = teacher_batch.select_idxs(np.asarray(bank_indices, dtype=np.int64))
            self._register_fixed_g_refs(teacher_for_bank)
        fixed_metrics = self._compute_fixed_g_probe_metrics()
        if not bool(self.skill_sd_cfg.get("enable_internalization", False)):
            return student_batch, {
                "skill_int/aux_skipped_internalization_disabled": 1.0,
                **fixed_metrics,
            }
        lambda_int_base = float(self.skill_sd_cfg.get("lambda_int", 0.0))
        warmup_steps = max(int(self.skill_sd_cfg.get("lambda_int_warmup_steps", 0)), 0)
        if warmup_steps > 0:
            scale = float(min(max(int(getattr(self, "global_steps", 0)), 0), warmup_steps)) / float(warmup_steps)
        else:
            scale = 1.0
        lambda_int = lambda_int_base * scale
        lambda_cf = self._cf_loss_weight()
        lambda_skill = self._skill_loss_weight()
        skill_loss_cfg = self._skill_loss_cfg()
        skill_loss_scope = str(skill_loss_cfg.get("scope", "successful_trajectories"))
        if lambda_int <= 0 and lambda_cf <= 0 and lambda_skill <= 0:
            return student_batch, {
                "skill_int/aux_skipped_lambda_zero": 1.0,
                "skill_int/lambda_int_warmup_scale": scale,
                **fixed_metrics,
            }

        self._ensure_skill_int_aux_tensors(student_batch)
        for key, default in (
            ("skill_int_cf_pair_id", None),
            ("skill_int_cf_student_row", -1),
            ("skill_int_cf_student_traj_uid", None),
            ("skill_int_cf_divergence_turn", -1),
            ("skill_int_cf_utility", 0.0),
            ("skill_int_cf_teacher_task_id", None),
            ("skill_int_cf_student_task_id", None),
        ):
            if key not in student_batch.non_tensor_batch:
                student_batch.non_tensor_batch[key] = np.asarray([default] * len(student_batch.batch), dtype=object)

        active_teacher = None
        active_teacher_log_probs = None
        active_quality_weights = None
        aux_proto = None
        cf_pair_proto = None
        cf_pair_records = []
        self._last_cf_pair_resolution_metrics = {}
        active_no_skill_missing = False
        if active_indices:
            active_teacher = teacher_batch.select_idxs(np.asarray(active_indices, dtype=np.int64))
            active_quality_weights = self._teacher_aux_quality_weights(active_teacher)
            if self._build_teacher_aux_no_skill_tensors(active_teacher) is None:
                active_no_skill_missing = True
            else:
                active_teacher_log_probs = self._compute_log_probs_for_proto(active_teacher)
                aux_proto = self._make_teacher_aux_proto(
                    student_batch=student_batch,
                    teacher_batch=active_teacher,
                    teacher_log_probs=active_teacher_log_probs,
                    utilities=utilities,
                    quality_weights=active_quality_weights,
                )
                if aux_proto is None:
                    active_no_skill_missing = True

        needs_cf_records = (
            aux_proto is not None
            and (lambda_cf > 0 or (lambda_skill > 0 and skill_loss_scope == "first_divergence"))
        )
        if needs_cf_records:
            if lambda_cf > 0:
                cf_pair_proto, cf_pair_records = self._make_cf_pair_aux_proto(
                    student_batch, active_teacher, aux_proto, active_indices
                )
            else:
                cf_pair_records = self._valid_cf_pair_records(student_batch, active_teacher, active_indices)

        skill_proto = None
        skill_records = []
        if lambda_skill > 0:
            if skill_loss_scope == "first_divergence":
                if active_teacher is not None and active_teacher_log_probs is not None:
                    skill_proto, skill_records = self._make_skill_imitation_aux_proto(
                        student_batch=student_batch,
                        active_teacher=active_teacher,
                        teacher_log_probs=active_teacher_log_probs,
                        utilities=utilities,
                        cf_pair_records=cf_pair_records,
                    )
            elif skill_indices:
                skill_teacher = teacher_batch.select_idxs(np.asarray(skill_indices, dtype=np.int64))
                skill_teacher_log_probs = self._compute_log_probs_for_proto(skill_teacher)
                skill_proto, skill_records = self._make_skill_imitation_aux_proto(
                    student_batch=student_batch,
                    active_teacher=skill_teacher,
                    teacher_log_probs=skill_teacher_log_probs,
                    utilities=skill_utilities,
                    cf_pair_records=[],
                )
            elif skill_loss_scope not in {"best_trajectory", "successful_trajectories"}:
                raise ValueError(f"Unknown skill_sd.skill_loss.scope: {skill_loss_scope}")

        state_query_batch = getattr(self, "_last_skill_int_state_query_batch", None)
        state_query_aux_proto = None
        state_query_no_skill_missing = False
        if lambda_int > 0 and state_query_batch is not None and len(state_query_batch.batch) > 0:
            if self._build_teacher_aux_no_skill_tensors(state_query_batch) is None:
                state_query_no_skill_missing = True
            else:
                state_query_log_probs = self._compute_log_probs_for_proto(state_query_batch)
                state_query_utilities = list(
                    state_query_batch.meta_info.get("skill_int_state_query_utilities", [0.0] * len(state_query_batch.batch))
                )
                state_query_qualities = list(
                    state_query_batch.meta_info.get("skill_int_state_query_qualities", [1.0] * len(state_query_batch.batch))
                )
                state_query_aux_proto = self._make_teacher_aux_proto(
                    student_batch=student_batch,
                    teacher_batch=state_query_batch,
                    teacher_log_probs=state_query_log_probs,
                    utilities=state_query_utilities,
                    quality_weights=state_query_qualities,
                )
                if state_query_aux_proto is not None:
                    query_scale = float(self._state_teacher_query_cfg().get("loss_scale", 1.0))
                    state_query_aux_proto.batch["skill_int_group_weight"] *= query_scale
                    aux_proto = (
                        DataProto.concat([aux_proto, state_query_aux_proto])
                        if aux_proto is not None
                        else state_query_aux_proto
                    )
                else:
                    state_query_no_skill_missing = True

        if aux_proto is None and cf_pair_proto is None and skill_proto is None:
            metrics = {
                "skill_int/aux_rows": 0.0,
                "skill_int/aux_active_groups": 0.0,
                "skill_int/skill_rows": 0.0,
                "skill_int/skill_candidate_rows": float(len(skill_indices)),
                "skill_int/skill_candidate_groups": 0.0,
                **fixed_metrics,
            }
            if active_no_skill_missing:
                metrics["skill_int/aux_skipped_no_alt_prompt"] = 1.0
            if state_query_no_skill_missing:
                metrics["skill_int/state_query_aux_skipped_no_alt_prompt"] = 1.0
            return student_batch, metrics

        if aux_proto is None and (lambda_int > 0 or lambda_cf > 0) and active_no_skill_missing:
            # Keep L_skill if available, but expose why L_int/L_cf were skipped.
            fixed_metrics = {**fixed_metrics, "skill_int/aux_skipped_no_alt_prompt": 1.0}

        if aux_proto is None and lambda_int > 0 and active_indices and skill_proto is None:
            return student_batch, {
                "skill_int/aux_skipped_no_alt_prompt": 1.0,
                **fixed_metrics,
            }

        world_size = max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1)
        probe_metrics = self._dump_cf_span_pair_probes(cf_pair_proto, cf_pair_records)
        student_chunks = student_batch.chunk(world_size)
        aux_chunks, pad_aux = self._build_worker_aux_chunks(aux_proto, cf_pair_proto, skill_proto, world_size)
        combined = DataProto.concat([
            DataProto.concat([student_chunk, aux_chunk])
            for student_chunk, aux_chunk in zip(student_chunks, aux_chunks)
        ])
        combined.meta_info["global_token_num"] = torch.sum(combined.batch["attention_mask"], dim=-1).tolist()
        combined.meta_info["skill_int_lambda_int"] = lambda_int
        combined.meta_info["skill_int_alpha"] = float(self.skill_sd_cfg.get("alpha", 1.0))
        combined.meta_info["skill_int_beta"] = float(self.skill_sd_cfg.get("beta", 1.0))
        combined.meta_info["skill_int_token_gate_beta"] = float(self.skill_sd_cfg.get("token_gate_beta", 1.0))
        combined.meta_info["skill_int_disable_sample_gate"] = bool(self.skill_sd_cfg.get("disable_sample_gate", False))
        combined.meta_info["skill_int_disable_token_gate"] = bool(self.skill_sd_cfg.get("disable_token_gate", False))
        entropy_gate_cfg = self.skill_sd_cfg.get("entropy_gate", {}) or {}
        combined.meta_info["skill_int_entropy_gate_enabled"] = bool(entropy_gate_cfg.get("enabled", False))
        combined.meta_info["skill_int_entropy_gate_threshold"] = entropy_gate_cfg.get("threshold", None)
        combined.meta_info["skill_int_entropy_gate_quantile"] = float(entropy_gate_cfg.get("quantile", 0.75))
        combined.meta_info["skill_int_entropy_gate_beta"] = float(entropy_gate_cfg.get("beta", 1.0))
        combined.meta_info["skill_int_entropy_gate_min"] = float(entropy_gate_cfg.get("min_gate", 0.0))
        combined.meta_info["skill_int_lambda_cf"] = float(lambda_cf)
        combined.meta_info["skill_int_beta_cf"] = float(self.skill_sd_cfg.get("beta_cf", 1.0))
        combined.meta_info["skill_int_g_thresh"] = float(self.skill_sd_cfg.get("g_thresh", 0.0))
        combined.meta_info["skill_int_cf_preference_mode"] = str(
            self.skill_sd_cfg.get("cf_pair_mode", "teacher_better")
        )
        combined.meta_info["skill_int_causal_probe_mode"] = _normalize_causal_probe_mode(
            self.skill_sd_cfg.get("causal_probe_mode", "off")
        )
        combined.meta_info["skill_int_causal_probe_eta"] = float(self.skill_sd_cfg.get("causal_probe_eta", 1.0))
        action_marker_token_ids, action_skip_token_ids = self._action_probe_token_ids()
        combined.meta_info["skill_int_action_marker_token_ids"] = action_marker_token_ids
        combined.meta_info["skill_int_action_skip_token_ids"] = action_skip_token_ids
        self._set_rescue_action_only_meta(combined)
        combined.meta_info["skill_int_lambda_skill"] = float(lambda_skill)
        combined.meta_info["skill_int_skill_loss_mode"] = str(skill_loss_cfg.get("mode", "ppo_sil"))
        combined.meta_info["skill_int_skill_clip_ratio"] = float(skill_loss_cfg.get("clip_ratio", 0.2))
        combined.meta_info["skill_int_skill_alpha"] = float(skill_loss_cfg.get("alpha", 1.0))
        combined.meta_info["skill_int_skill_use_utility_weight"] = bool(skill_loss_cfg.get("use_utility_weight", True))

        active_groups = set()
        if active_teacher is not None:
            active_groups = {
                str(active_teacher.non_tensor_batch["uid"][i])
                for i in range(len(active_teacher.non_tensor_batch.get("uid", [])))
            }
        if state_query_aux_proto is not None:
            active_groups.update(
                str(state_query_aux_proto.non_tensor_batch["uid"][i])
                for i in range(len(state_query_aux_proto.non_tensor_batch.get("uid", [])))
            )
        skill_groups = set()
        if skill_proto is not None:
            skill_groups = {
                str(skill_proto.non_tensor_batch["uid"][i])
                for i in range(len(skill_proto.non_tensor_batch.get("uid", [])))
                if skill_proto.non_tensor_batch["uid"][i] is not None
            }
        fixed_count = float(aux_proto.batch["skill_int_fixed_ref_mask"].sum().item()) if aux_proto is not None else 0.0
        teacher_aux_rows = float(len(active_teacher) if active_teacher is not None else 0)
        state_query_rows = float(len(state_query_aux_proto.batch) if state_query_aux_proto is not None else 0)
        return combined, {
            "skill_int/aux_rows": teacher_aux_rows + state_query_rows,
            "skill_int/teacher_aux_rows": teacher_aux_rows,
            "skill_int/teacher_aux_quality_mean": (
                float(active_quality_weights.mean().item())
                if active_quality_weights is not None and active_quality_weights.numel() else 0.0
            ),
            "skill_int/teacher_aux_quality_positive_frac": (
                float((active_quality_weights > 0).to(torch.float32).mean().item())
                if active_quality_weights is not None and active_quality_weights.numel() else 0.0
            ),
            "skill_int/state_query_aux_rows": state_query_rows,
            "skill_int/state_query_loss_scale": float(self._state_teacher_query_cfg().get("loss_scale", 1.0)),
            "skill_int/state_query_aux_skipped_no_alt_prompt": float(state_query_no_skill_missing),
            "skill_int/aux_rows_padded": float(pad_aux),
            "skill_int/aux_active_groups": float(len(active_groups)),
            "skill_int/cf_candidate_pairs_attached": float(len(cf_pair_records)),
            "skill_int/cf_aux_student_rows": float(len(cf_pair_records)),
            "skill_int/skill_rows": float(len(skill_proto.batch) if skill_proto is not None else 0),
            "skill_int/skill_candidate_rows": float(len(skill_indices)),
            "skill_int/skill_candidate_groups": float(len(skill_groups)),
            "skill_int/skill_scope_first_divergence": float(skill_loss_scope == "first_divergence"),
            "skill_int/skill_scope_successful_trajectories": float(skill_loss_scope == "successful_trajectories"),
            "skill_int/skill_gate_teacher_success": float(str(skill_loss_cfg.get("gate", "teacher_success")) == "teacher_success"),
            "skill_int/lambda_int": lambda_int,
            "skill_int/lambda_cf": float(lambda_cf),
            "skill_int/lambda_skill": float(lambda_skill),
            "skill_int/lambda_int_warmup_scale": scale,
            "skill_int/normalize_aux_by_group": float(bool(self.skill_sd_cfg.get("normalize_aux_by_group", True))),
            "skill_int/fixed_ref_tasks": float(len(self._skill_int_fixed_ref_ids)),
            "skill_int/fixed_ref_aux_rows": fixed_count,
            **self._last_cf_pair_resolution_metrics,
            **fixed_metrics,
            **probe_metrics,
        }

    def _install_student_first_metric_hook(self) -> None:
        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        if actor_rollout_wg is None:
            return
        orig_upd = getattr(actor_rollout_wg, "_skill_int_orig_update_actor", actor_rollout_wg.update_actor)
        actor_rollout_wg._skill_int_orig_update_actor = orig_upd

        def update_actor_wrapper(batch):
            update_batch, aux_metrics = self._attach_teacher_aux_batch(batch)
            if aux_metrics:
                prev = update_batch.meta_info.get("skill_int_metrics", {})
                prev.update(aux_metrics)
                update_batch.meta_info["skill_int_metrics"] = prev
            self._last_skill_int_teacher_batch = None
            self._last_skill_int_state_query_batch = None
            self._last_skill_int_cf_branch_proto = None
            self._last_skill_int_prefix_rescue_proto = None
            self._last_skill_int_prefix_rescue_suffix_proto = None

            from recipe.skill_sd.paper_objective import prepare_weights
            prepare_weights(update_batch, self._prefix_rescue_action_mask)
            actor_output = orig_upd(update_batch)
            meta_info = getattr(update_batch, "meta_info", {}) or {}
            skill_int_metrics = dict(meta_info.get("skill_int_metrics", {}))
            existing = dict(actor_output.meta_info.get("metrics") or {})
            actor_skill_int_metrics = {
                key: value for key, value in existing.items() if isinstance(key, str) and key.startswith("skill_int/")
            }
            if actor_skill_int_metrics:
                skill_int_metrics.update(actor_skill_int_metrics)
            if skill_int_metrics:
                logged_metrics = {
                    key: value for key, value in existing.items() if not (isinstance(key, str) and key.startswith("skill_int/"))
                }
                logged_metrics.update(self._compact_cpsd_wandb_metrics(skill_int_metrics))
                for metric_source in (skill_int_metrics, existing):
                    for key, value in metric_source.items():
                        if isinstance(key, str) and key.startswith("prefix_rescue/"):
                            suffix = key.split("/", 1)[1]
                            logged_metrics[key] = value
                            logged_metrics[f"rescue/{suffix}"] = value
                if self._skill_int_wandb_debug_enabled():
                    logged_metrics.update(skill_int_metrics)
                actor_output.meta_info["metrics"] = logged_metrics
            self._log_actor_aux_dynamics(update_batch, skill_int_metrics)
            return actor_output

        actor_rollout_wg.update_actor = update_actor_wrapper
        print("[SkillSDTrainer] student-first auxiliary loss hook installed.")

    def _skill_int_wandb_debug_enabled(self) -> bool:
        logging_cfg = self.skill_sd_cfg.get("logging", {}) if self.skill_sd_cfg is not None else {}
        return bool((logging_cfg or {}).get("wandb_debug", False))

    def _compact_cpsd_wandb_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        return {
            target: metrics[source]
            for source, target in CPSD_WANDB_METRIC_MAP.items()
            if source in metrics
        }

    # ------------------------------------------------------------------ #
    # Experiment evidence hooks
    # ------------------------------------------------------------------ #

    def _log_intervention_dynamic_csv(self, student_stats, decisions, metrics) -> None:
        path = self._resolve_logging_path("intervention_csv", "intervention_dynamic.csv")
        if path is None:
            return
        if not decisions:
            self._append_csv(path, {
                "global_step": int(getattr(self, "global_steps", 0)),
                "group_uid": "__aggregate__",
                "student_reward_mean": metrics.get("skill_int/student_reward_mean", 0.0),
                "teacher_used": 0,
                "teacher_success": 0,
                "utility": 0.0,
                "aux_active": 0,
                "teacher_candidate_groups": metrics.get("skill_int/teacher_candidate_groups", 0.0),
                "teacher_trigger_reward_mean_threshold": metrics.get("skill_int/teacher_trigger_reward_mean_threshold", ""),
                "active_teacher_groups": metrics.get("skill_int/active_teacher_groups", 0.0),
                "G_mean": "",
                "fixed_G_mean": "",
                "fixed_with_skill_logp_mean": "",
                "fixed_no_skill_logp_mean": "",
                "fixed_G_count": "",
                "fixed_G_task_count": "",
                "skill_rows": "",
                "skill_loss": "",
                "aux_rows": "",
            }, fieldnames=INTERVENTION_CSV_FIELDS)
            return
        teacher_cfg = self._teacher_rollout_cfg() or {}
        aux_gate_mode = str(teacher_cfg.get("aux_gate_mode", "success"))
        utility_eps = float(teacher_cfg.get("utility_eps", 0.0))
        for group_uid, decision in decisions.items():
            aux_active = _teacher_aux_gate_active(
                aux_gate_mode,
                bool(decision.teacher_success),
                float(decision.utility),
                utility_eps,
            )
            stats = student_stats.get(group_uid)
            self._append_csv(path, {
                "global_step": int(getattr(self, "global_steps", 0)),
                "group_uid": group_uid,
                "student_reward_mean": stats.reward_mean if stats else "",
                "student_reward_best": stats.reward_best if stats else "",
                "student_success_frac": stats.success_frac if stats else "",
                "teacher_used": int(decision.teacher_used),
                "teacher_success": int(decision.teacher_success),
                "utility": float(decision.utility),
                "utility_mode": decision.utility_mode,
                "aux_active": int(aux_active),
                "teacher_reward_best": float(decision.teacher_reward_best),
                "teacher_task_score_best": float(decision.teacher_task_score_best),
                "student_task_score_mean": float(decision.student_task_score_mean),
                "teacher_candidate_groups": metrics.get("skill_int/teacher_candidate_groups", 0.0),
                "teacher_trigger_reward_mean_threshold": metrics.get("skill_int/teacher_trigger_reward_mean_threshold", ""),
                "active_teacher_groups": metrics.get("skill_int/active_teacher_groups", 0.0),
                "G_mean": "",
                "fixed_G_mean": "",
                "fixed_with_skill_logp_mean": "",
                "fixed_no_skill_logp_mean": "",
                "fixed_G_count": "",
                "fixed_G_task_count": "",
                "skill_rows": "",
                "skill_loss": "",
                "aux_rows": "",
            }, fieldnames=INTERVENTION_CSV_FIELDS)

    def _log_actor_aux_dynamics(self, update_batch, metrics) -> None:
        path = self._resolve_logging_path("intervention_csv", "intervention_dynamic.csv")
        if path is None:
            return
        row = {
            "global_step": int(getattr(self, "global_steps", 0)),
            "group_uid": "__actor_aux__",
            "student_reward_mean": metrics.get("skill_int/student_reward_mean", ""),
            "student_reward_best": metrics.get("skill_int/student_reward_best_mean", ""),
            "student_success_frac": metrics.get("skill_int/student_success_frac", ""),
            "teacher_used": "",
            "teacher_success": "",
            "utility": "",
            "utility_mode": "",
            "aux_active": int(float(metrics.get("skill_int/aux_rows", 0.0)) > 0),
            "teacher_reward_best": metrics.get("skill_int/teacher_reward_best_mean", ""),
            "teacher_task_score_best": metrics.get("skill_int/teacher_task_score_best_mean", ""),
            "student_task_score_mean": "",
            "teacher_candidate_groups": metrics.get("skill_int/teacher_candidate_groups", ""),
            "teacher_trigger_reward_mean_threshold": metrics.get("skill_int/teacher_trigger_reward_mean_threshold", ""),
            "active_teacher_groups": metrics.get("skill_int/aux_active_groups", metrics.get("skill_int/active_teacher_groups", "")),
            "G_mean": metrics.get("skill_int/G_mean", ""),
            "fixed_G_mean": metrics.get("skill_int/fixed_G_mean", ""),
            "fixed_with_skill_logp_mean": metrics.get("skill_int/fixed_with_skill_logp_mean", ""),
            "fixed_no_skill_logp_mean": metrics.get("skill_int/fixed_no_skill_logp_mean", ""),
            "fixed_G_count": metrics.get("skill_int/fixed_G_count", ""),
            "fixed_G_task_count": metrics.get("skill_int/fixed_G_task_count", ""),
            "skill_rows": metrics.get("skill_int/skill_rows", ""),
            "skill_loss": metrics.get("skill_int/skill_loss", ""),
            "aux_rows": metrics.get("skill_int/aux_rows", ""),
        }
        self._append_csv(path, row, fieldnames=INTERVENTION_CSV_FIELDS)
        fixed_path = self._resolve_logging_path("fixed_g_jsonl", "fixed_g_records.jsonl")
        if fixed_path is not None and ("skill_int/fixed_G_mean" in metrics or self._fixed_g_enabled()):
            record = dict(getattr(self, "_last_fixed_g_record", None) or {})
            record.update({
                "global_step": int(getattr(self, "global_steps", 0)),
                "fixed_ref_ids": list(self._skill_int_fixed_ref_order),
                "fixed_G_mean": metrics.get("skill_int/fixed_G_mean"),
                "fixed_G_std": metrics.get("skill_int/fixed_G_std"),
                "fixed_with_skill_logp_mean": metrics.get("skill_int/fixed_with_skill_logp_mean"),
                "fixed_no_skill_logp_mean": metrics.get("skill_int/fixed_no_skill_logp_mean"),
                "fixed_G_count": metrics.get("skill_int/fixed_G_count", 0.0),
                "fixed_G_task_count": metrics.get("skill_int/fixed_G_task_count", 0.0),
                "fixed_G_token_count": metrics.get("skill_int/fixed_G_token_count", 0.0),
                "G_mean_active_aux": metrics.get("skill_int/G_mean"),
                "fixed_aux_G_mean": metrics.get("skill_int/fixed_aux_G_mean"),
                "fixed_aux_G_count": metrics.get("skill_int/fixed_aux_G_count", 0.0),
                "aux_rows": metrics.get("skill_int/aux_rows"),
                "skill_rows": metrics.get("skill_int/skill_rows"),
                "skill_loss": metrics.get("skill_int/skill_loss"),
            })
            self._append_jsonl(fixed_path, record)

    def _log_task_success_trajectories(self, batch, *, arm: str, decisions=None) -> None:
        path = self._resolve_logging_path("task_success_jsonl", "task_success_trajectories.jsonl")
        if path is None:
            return
        fields = batch.non_tensor_batch
        traj_values = np.asarray(fields.get("traj_uid", []), dtype=object)
        seen = set()
        threshold = float((self._teacher_rollout_cfg() or {}).get("success_reward_threshold", 10.0))
        for i in range(len(batch.batch)):
            traj_uid = str(traj_values[i]) if len(traj_values) else f"row-{i}"
            if traj_uid in seen:
                continue
            seen.add(traj_uid)
            group_uid = str(self._field_at(fields, "uid", i, ""))
            decision = decisions.get(group_uid) if decisions else None
            episode_reward = self._float_value(self._field_at(fields, "episode_rewards", i, 0.0))
            self._append_jsonl(path, {
                "global_step": int(getattr(self, "global_steps", 0)),
                "arm": arm,
                "task_id": self._task_identity(fields, i),
                "uid": group_uid,
                "traj_uid": traj_uid,
                "episode_reward": episode_reward,
                "episode_task_score": self._float_value(self._field_at(fields, "episode_task_scores", i, 0.0)),
                "episode_length": self._float_value(self._field_at(fields, "episode_lengths", i, 0.0)),
                "success": bool(episode_reward >= threshold),
                "teacher_success": bool(decision.teacher_success) if decision is not None else None,
                "utility": float(decision.utility) if decision is not None else None,
            })

    # ------------------------------------------------------------------ #
    # Validation arms
    # ------------------------------------------------------------------ #

    def _validate(self):
        base_validate = super()._validate
        cfg = self.skill_sd_cfg
        val_envs = getattr(self, "val_envs", None)
        has_skill_mem = val_envs is not None and getattr(val_envs, "retrieval_memory", None) is not None
        mode = (cfg.get("val_arm_mode", None) if cfg is not None else None) or (
            "both" if (cfg is not None and bool(cfg.get("enable_internalization", False))) else "teacher"
        )
        if not has_skill_mem or mode == "teacher":
            self._restart_alfworld_val_game_sequences()
            self._val_arm_tag = "teacher" if has_skill_mem else "default"
            alt_ctx = (
                _force_teacher_alt_prompt_tensors(
                    val_envs, self.traj_collector, _infer_env_num(val_envs)
                )
                if has_skill_mem
                else nullcontext()
            )
            with self._fixed_webshop_val_sessions(), alt_ctx:
                metrics = base_validate() or {}
            self._val_arm_tag = "default"
            return metrics
        if mode == "student":
            print("[SkillSDTrainer] val_arm_mode=student: running no-skill prompt validation")
            self._restart_alfworld_val_game_sequences()
            self._val_arm_tag = "student"
            with self._fixed_webshop_val_sessions(), _disable_retrieval_memory(val_envs):
                metrics = base_validate() or {}
            self._val_arm_tag = "default"
            return metrics
        if mode != "both":
            raise ValueError(f"Unknown skill_sd.val_arm_mode: {mode}")

        teacher_val_freq_raw = cfg.get("teacher_val_freq", None)
        teacher_val_freq = None
        if teacher_val_freq_raw is not None:
            try:
                teacher_val_freq = int(teacher_val_freq_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "skill_sd.teacher_val_freq must be a positive integer or null"
                ) from exc
            if teacher_val_freq <= 0:
                raise ValueError(
                    "skill_sd.teacher_val_freq must be a positive integer or null"
                )

        if teacher_val_freq is not None and self.global_steps % teacher_val_freq != 0:
            print(
                "[SkillSDTrainer] val_arm_mode=both: skipping teacher validation "
                f"at step {self.global_steps} (teacher_val_freq={teacher_val_freq}); "
                "running student validation only"
            )
            self._restart_alfworld_val_game_sequences()
            self._val_arm_tag = "student"
            with self._fixed_webshop_val_sessions(), _disable_retrieval_memory(val_envs):
                student_metrics = base_validate() or {}
            self._val_arm_tag = "default"
            merged = _prefix_metrics("student", student_metrics)
            merged["val/teacher_skipped"] = 1.0
            merged["val/teacher_val_freq"] = float(teacher_val_freq)
            return merged

        print("[SkillSDTrainer] val_arm_mode=both: running teacher validation")
        paired_alfworld_val = self._restart_alfworld_val_game_sequences()
        self._val_arm_tag = "teacher"
        teacher_alt_ctx = _force_teacher_alt_prompt_tensors(
            val_envs, self.traj_collector, _infer_env_num(val_envs)
        )
        with self._fixed_webshop_val_sessions(), teacher_alt_ctx:
            teacher_metrics = base_validate() or {}
        print("[SkillSDTrainer] val_arm_mode=both: running student validation")
        if self._restart_alfworld_val_game_sequences() != paired_alfworld_val:
            raise RuntimeError("ALFWorld validation sequence reset changed between arms")
        self._val_arm_tag = "student"
        with self._fixed_webshop_val_sessions(), _disable_retrieval_memory(val_envs):
            student_metrics = base_validate() or {}
        self._val_arm_tag = "default"

        alpha = float(cfg.get("alpha", 1.0))
        merged: Dict[str, float] = {}
        merged.update(_prefix_metrics("teacher", teacher_metrics))
        merged.update(_prefix_metrics("student", student_metrics))
        merged.update(_compute_utility(teacher_metrics, student_metrics, alpha=alpha))
        if teacher_val_freq is not None:
            merged["val/teacher_skipped"] = 0.0
            merged["val/teacher_val_freq"] = float(teacher_val_freq)
        if paired_alfworld_val:
            merged["val/paired_game_sequence"] = 1.0
        return merged
