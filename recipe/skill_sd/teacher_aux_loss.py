"""Auxiliary losses for CPSD / Skill-SD teacher rows.

Teacher rows are off-policy demonstrations. They must not contribute to
PPO/GRPO advantages or the PPO policy-gradient term; this module only computes
losses for rows explicitly marked as auxiliary rows.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

try:
    from verl.trainer.ppo.core_algos import agg_loss
except ModuleNotFoundError:
    def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
        if loss_agg_mode == "token-mean":
            weights = loss_mask.to(loss_mat.dtype)
            return (loss_mat * weights).sum() / weights.sum().clamp(min=1.0)
        if loss_agg_mode == "seq-mean-token-sum":
            return torch.sum(loss_mat * loss_mask, dim=-1).mean()
        if loss_agg_mode == "seq-mean-token-mean":
            weights = loss_mask.to(loss_mat.dtype)
            seq_losses = torch.sum(loss_mat * weights, dim=-1) / weights.sum(dim=-1).clamp(min=1.0)
            return seq_losses.mean()
        if loss_agg_mode == "seq-mean-token-sum-norm":
            return torch.sum(loss_mat * loss_mask, dim=-1).sum() / loss_mask.shape[-1]
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def _sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)


def _first_probe_token_indices(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    marker_token_ids: Sequence[int] | None = None,
    skip_token_ids: Sequence[int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the first action-content token for one-step causal probing.

    If ``marker_token_ids`` is absent or not found in a row, fall back to the
    first valid response token to preserve the old behavior.
    """
    first_valid_idx = response_mask.float().argmax(dim=-1).clamp(min=0).long()
    found = torch.zeros_like(first_valid_idx, dtype=torch.bool)
    if responses.numel() == 0 or not marker_token_ids:
        return first_valid_idx, found

    marker = [int(x) for x in marker_token_ids]
    marker_len = len(marker)
    if marker_len <= 0 or marker_len >= responses.shape[-1]:
        return first_valid_idx, found
    skip_set = {int(x) for x in (skip_token_ids or [])}
    marker_t = torch.as_tensor(marker, device=responses.device, dtype=responses.dtype)

    probe_idx = first_valid_idx.clone()
    response_mask_bool = response_mask.bool()
    for row in range(responses.shape[0]):
        row_tokens = responses[row]
        valid = response_mask_bool[row]
        valid_positions = torch.nonzero(valid, as_tuple=False).view(-1)
        for pos_t in valid_positions:
            pos = int(pos_t.item())
            end = pos + marker_len
            if end >= responses.shape[-1]:
                continue
            if not torch.equal(row_tokens[pos:end], marker_t):
                continue
            content_pos = end
            while (
                content_pos < responses.shape[-1]
                and bool(valid[content_pos].item())
                and int(row_tokens[content_pos].item()) in skip_set
            ):
                content_pos += 1
            if content_pos < responses.shape[-1] and bool(valid[content_pos].item()):
                probe_idx[row] = int(content_pos)
                found[row] = True
            break
    return probe_idx, found


def action_span_mask(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    start_marker_ids: Sequence[int] | None,
    end_marker_ids: Sequence[int] | None,
    marker_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]] | None = None,
    missing_marker_policy: str = "full",
) -> torch.Tensor:
    """Keep only tokens inside the first recognized decision span per response.

    ``marker_pairs`` supports environments with multiple decision tags, such as
    SearchQA's ``<search>`` and ``<answer>``. The legacy start/end arguments are
    retained for callers that use a single ``<action>`` span.

    ``missing_marker_policy='full'`` preserves the legacy full-response fallback.
    Strict callers may pass ``'zero'`` so malformed rows are completely inert.
    """
    if missing_marker_policy not in {"full", "zero"}:
        raise ValueError(
            f"Unknown missing_marker_policy: {missing_marker_policy}"
        )
    out = (
        response_mask.clone()
        if missing_marker_policy == "full"
        else torch.zeros_like(response_mask)
    )
    L = responses.shape[-1]
    raw_pairs = list(marker_pairs or [])
    if not raw_pairs and start_marker_ids:
        raw_pairs = [(start_marker_ids, end_marker_ids or [])]
    pairs = [
        ([int(x) for x in start], [int(x) for x in end])
        for start, end in raw_pairs
        if start
    ]
    if responses.numel() == 0 or not pairs:
        return out
    token_pairs = [
        (
            start,
            end,
            torch.as_tensor(start, device=responses.device, dtype=responses.dtype),
            torch.as_tensor(end, device=responses.device, dtype=responses.dtype) if end else None,
        )
        for start, end in pairs
        if len(start) < L
    ]
    if not token_pairs:
        return out
    valid_bool = response_mask.bool()
    for row in range(responses.shape[0]):
        toks = responses[row]
        match = None
        for start, end, start_t, end_t in token_pairs:
            for pos in range(L - len(start) + 1):
                if not bool(valid_bool[row, pos:pos + len(start)].all().item()):
                    continue
                if torch.equal(toks[pos:pos + len(start)], start_t):
                    candidate = (pos, pos + len(start), end, end_t)
                    if match is None or pos < match[0]:
                        match = candidate
                    break
        if match is None:
            continue  # keep full-response fallback for this row
        _, s_pos, end, end_t = match
        e_pos = L
        end_found = end_t is None
        if end_t is not None:
            for pos in range(s_pos, L - len(end) + 1):
                if (
                    bool(valid_bool[row, pos:pos + len(end)].all().item())
                    and torch.equal(toks[pos:pos + len(end)], end_t)
                ):
                    e_pos = pos
                    end_found = True
                    break
        if not end_found:
            continue
        keep = torch.zeros(L, dtype=out.dtype, device=out.device)
        keep[s_pos:e_pos] = 1
        out[row] = response_mask[row] * keep
    return out


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return float(value)


def _entropy_exploration_gate(
    *,
    entropy: torch.Tensor | None,
    response_mask: torch.Tensor,
    enabled: bool = False,
    threshold: float | None = None,
    quantile: float = 0.75,
    beta: float = 1.0,
    min_gate: float = 0.0,
    metric_prefix: str = "skill_int/entropy",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a stop-gradient gate that suppresses distillation on high-entropy actions."""
    gate_dtype = response_mask.dtype if response_mask.is_floating_point() else torch.float32
    gate = torch.ones_like(response_mask, dtype=gate_dtype)
    metrics = {
        f"{metric_prefix}_gate_enabled": float(enabled),
        f"{metric_prefix}_mean": 0.0,
        f"{metric_prefix}_threshold": 0.0,
        f"{metric_prefix}_gate_mean": 1.0,
        f"{metric_prefix}_gate_min": 1.0,
        f"{metric_prefix}_suppressed_frac": 0.0,
    }
    if not enabled or entropy is None:
        return gate, metrics

    entropy = entropy.detach().to(gate.dtype)
    mask_bool = response_mask.bool()
    valid_entropy = entropy[mask_bool]
    if valid_entropy.numel() == 0:
        return gate, metrics

    threshold_value = _float_or_none(threshold)
    if threshold_value is None:
        q = min(max(float(quantile), 0.0), 1.0)
        threshold_t = torch.quantile(valid_entropy.float(), q).to(device=entropy.device, dtype=entropy.dtype)
    else:
        threshold_t = torch.as_tensor(threshold_value, device=entropy.device, dtype=entropy.dtype)

    beta = max(float(beta), 0.0)
    min_gate = min(max(float(min_gate), 0.0), 1.0)
    excess = (entropy - threshold_t).clamp(min=0.0)
    if beta == 0.0:
        gate = torch.ones_like(entropy, dtype=gate.dtype)
    else:
        gate = min_gate + (1.0 - min_gate) * torch.exp(-beta * excess)
    gate = gate.detach().to(gate_dtype)

    valid_gate = gate[mask_bool]
    metrics.update({
        f"{metric_prefix}_mean": float(valid_entropy.mean().detach().cpu()),
        f"{metric_prefix}_threshold": float(threshold_t.detach().cpu()),
        f"{metric_prefix}_gate_mean": float(valid_gate.mean().detach().cpu()),
        f"{metric_prefix}_gate_min": float(valid_gate.min().detach().cpu()),
        f"{metric_prefix}_suppressed_frac": float((valid_gate < 0.999).to(torch.float32).mean().detach().cpu()),
    })
    return gate, metrics


def compute_teacher_internalization_loss(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    utility: torch.Tensor,
    group_weight: torch.Tensor | None = None,
    student_entropy: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    token_gate_beta: float = 1.0,
    disable_sample_gate: bool = False,
    disable_token_gate: bool = False,
    entropy_gate_enabled: bool = False,
    entropy_gate_threshold: float | None = None,
    entropy_gate_quantile: float = 0.75,
    entropy_gate_beta: float = 1.0,
    entropy_gate_min: float = 0.0,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute gated sampled-action KL surrogate for teacher demos."""
    if response_mask.sum().detach().item() <= 0:
        zero = student_log_probs.sum() * 0.0
        return zero, {
            "skill_int/int_loss": 0.0,
            "skill_int/G_mean": 0.0,
            "skill_int/G_std": 0.0,
            "skill_int/W_mean": 0.0,
            "skill_int/W_std": 0.0,
            "skill_int/token_gate_mean": 0.0,
            "skill_int/aux_tokens": 0.0,
            "skill_int/int_group_weight_mean": 0.0,
            "skill_int/disable_sample_gate": float(disable_sample_gate),
            "skill_int/disable_token_gate": float(disable_token_gate),
            "skill_int/entropy_gate_enabled": float(entropy_gate_enabled),
            "skill_int/entropy_mean": 0.0,
            "skill_int/entropy_threshold": 0.0,
            "skill_int/entropy_gate_mean": 1.0,
            "skill_int/entropy_gate_min": 1.0,
            "skill_int/entropy_suppressed_frac": 0.0,
        }

    teacher_log_probs = teacher_log_probs.detach()
    gap = teacher_log_probs - student_log_probs.detach()
    sample_g = _sample_mean(gap, response_mask)
    utility = utility.to(student_log_probs.dtype).view(-1)
    if group_weight is None:
        group_weight = torch.ones_like(utility)
    else:
        group_weight = group_weight.to(student_log_probs.dtype).view(-1).detach()
    if disable_sample_gate:
        weight = torch.ones_like(sample_g)
    else:
        weight = torch.sigmoid((alpha * utility + beta * sample_g).clamp(min=-30.0, max=30.0)).detach()
    if disable_token_gate:
        token_gate = torch.ones_like(gap)
    else:
        token_gate = torch.sigmoid((token_gate_beta * gap).clamp(min=-30.0, max=30.0)).detach()
    entropy_gate, entropy_metrics = _entropy_exploration_gate(
        entropy=student_entropy,
        response_mask=response_mask,
        enabled=entropy_gate_enabled,
        threshold=entropy_gate_threshold,
        quantile=entropy_gate_quantile,
        beta=entropy_gate_beta,
        min_gate=entropy_gate_min,
    )

    loss_mat = (
        group_weight.unsqueeze(-1)
        * weight.unsqueeze(-1)
        * token_gate
        * entropy_gate
        * (teacher_log_probs - student_log_probs)
    )
    loss = agg_loss(loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask = response_mask.to(student_log_probs.dtype)
        token_count = float(mask.sum().detach().cpu())
        gate_mean = float(((token_gate * mask).sum() / mask.sum().clamp(min=1.0)).detach().cpu())
        metrics = {
            "skill_int/int_loss": float(loss.detach().cpu()),
            "skill_int/G_mean": float(sample_g.mean().detach().cpu()),
            "skill_int/G_std": float(sample_g.std().detach().cpu()) if sample_g.numel() > 1 else 0.0,
            "skill_int/W_mean": float(weight.mean().detach().cpu()),
            "skill_int/W_std": float(weight.std().detach().cpu()) if weight.numel() > 1 else 0.0,
            "skill_int/token_gate_mean": gate_mean,
            "skill_int/aux_tokens": token_count,
            "skill_int/int_group_weight_mean": float(group_weight.mean().detach().cpu()),
            "skill_int/disable_sample_gate": float(disable_sample_gate),
            "skill_int/disable_token_gate": float(disable_token_gate),
        }
        metrics.update(entropy_metrics)
    return loss, metrics


def compute_skill_conditioned_imitation_loss(
    *,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    utility: torch.Tensor | None = None,
    group_weight: torch.Tensor | None = None,
    student_entropy: torch.Tensor | None = None,
    mode: str = "ppo_sil",
    clip_ratio: float = 0.2,
    alpha: float = 1.0,
    use_utility_weight: bool = True,
    entropy_gate_enabled: bool = False,
    entropy_gate_threshold: float | None = None,
    entropy_gate_quantile: float = 0.75,
    entropy_gate_beta: float = 1.0,
    entropy_gate_min: float = 0.0,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Keep the policy competent under the with-skill prompt on filtered teacher actions."""
    zero = log_probs.sum() * 0.0
    if response_mask.sum().detach().item() <= 0:
        return zero, {
            "skill_int/skill_loss": 0.0,
            "skill_int/skill_tokens": 0.0,
            "skill_int/skill_weight_mean": 0.0,
            "skill_int/skill_ratio_mean": 1.0,
            "skill_int/skill_clipfrac": 0.0,
            "skill_int/skill_entropy_gate_mean": 1.0,
            "skill_int/skill_group_weight_mean": 0.0,
        }

    mode = str(mode)
    old_log_probs = old_log_probs.detach()
    mask = response_mask.to(log_probs.dtype)
    if utility is None or not use_utility_weight:
        weight = torch.ones(log_probs.shape[0], device=log_probs.device, dtype=log_probs.dtype)
    else:
        utility = utility.to(log_probs.dtype).view(-1)
        weight = torch.sigmoid((float(alpha) * utility).clamp(min=-30.0, max=30.0)).detach()
    if group_weight is None:
        group_weight = torch.ones(log_probs.shape[0], device=log_probs.device, dtype=log_probs.dtype)
    else:
        group_weight = group_weight.to(log_probs.dtype).view(-1).detach()

    entropy_gate, entropy_metrics = _entropy_exploration_gate(
        entropy=student_entropy,
        response_mask=response_mask,
        enabled=entropy_gate_enabled,
        threshold=entropy_gate_threshold,
        quantile=entropy_gate_quantile,
        beta=entropy_gate_beta,
        min_gate=entropy_gate_min,
        metric_prefix="skill_int/skill_entropy",
    )

    if mode == "ppo_sil":
        ratio = torch.exp((log_probs - old_log_probs).clamp(min=-30.0, max=30.0))
        upper = 1.0 + float(clip_ratio)
        lower = max(1.0 - float(clip_ratio), 0.0)
        clipped = ratio.clamp(min=lower, max=upper)
        surrogate = torch.minimum(ratio, clipped)
        loss_mat = -group_weight.unsqueeze(-1) * weight.unsqueeze(-1) * entropy_gate * surrogate
    elif mode == "bc":
        ratio = torch.exp((log_probs - old_log_probs).detach().clamp(min=-30.0, max=30.0))
        loss_mat = -group_weight.unsqueeze(-1) * weight.unsqueeze(-1) * entropy_gate * log_probs
    else:
        raise ValueError(f"Unknown skill imitation loss mode: {mode}")

    loss = agg_loss(loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        valid = response_mask.bool()
        valid_ratio = ratio[valid] if valid.any() else ratio.new_ones((1,))
        valid_entropy_gate = entropy_gate[valid] if valid.any() else entropy_gate.new_ones((1,))
        metrics = {
            "skill_int/skill_loss": float(loss.detach().cpu()),
            "skill_int/skill_tokens": float(mask.sum().detach().cpu()),
            "skill_int/skill_weight_mean": float(weight.mean().detach().cpu()) if weight.numel() else 0.0,
            "skill_int/skill_ratio_mean": float(valid_ratio.mean().detach().cpu()),
            "skill_int/skill_clipfrac": float((valid_ratio > (1.0 + float(clip_ratio))).to(torch.float32).mean().detach().cpu()),
            "skill_int/skill_entropy_gate_mean": float(valid_entropy_gate.mean().detach().cpu()),
            "skill_int/skill_group_weight_mean": float(group_weight.mean().detach().cpu()),
        }
        metrics.update(entropy_metrics)
    return loss, metrics


def compute_rescue_suffix_imitation_loss(
    *,
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    group_weight: torch.Tensor | None = None,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Low-weight action-only imitation on a terminally verified suffix."""
    zero = log_probs.sum() * 0.0
    if response_mask.sum().detach().item() <= 0:
        return zero, {
            "prefix_rescue/suffix_loss": 0.0,
            "prefix_rescue/suffix_tokens": 0.0,
            "prefix_rescue/suffix_rows": 0.0,
        }
    if group_weight is None:
        group_weight = torch.ones(
            log_probs.shape[0], device=log_probs.device, dtype=log_probs.dtype
        )
    else:
        group_weight = group_weight.to(log_probs.dtype).view(-1).detach()
    loss = agg_loss(
        loss_mat=-group_weight.unsqueeze(-1) * log_probs,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    active_rows = response_mask.bool().any(dim=-1)
    return loss, {
        "prefix_rescue/suffix_loss": float(loss.detach().cpu()),
        "prefix_rescue/suffix_tokens": float(response_mask.sum().detach().cpu()),
        "prefix_rescue/suffix_rows": float(active_rows.sum().detach().cpu()),
    }


def compute_rescue_suffix_topk_kl(
    *,
    q_topk_log_probs: torch.Tensor,
    q_residual_mass: torch.Tensor,
    p_topk_log_probs: torch.Tensor,
    p_residual_mass: torch.Tensor,
    response_mask: torch.Tensor,
    group_weight: torch.Tensor | None = None,
    eps: float = 1.0e-8,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Top-k plus residual KL for skill-to-no-skill suffix transfer.

    The top-k token ids are selected by the with-skill distribution ``q`` and
    shared with ``p`` by the caller.  All q-side values are detached here as a
    second guard against accidentally training the privileged branch.
    """
    q_logp = q_topk_log_probs.detach()
    q_res = q_residual_mass.detach()
    mask = response_mask.to(p_topk_log_probs.dtype)
    zero = p_topk_log_probs.sum() * 0.0
    if mask.sum().detach().item() <= 0:
        return zero, {
            "prefix_rescue/suffix_kl": 0.0,
            "prefix_rescue/suffix_kl_tokens": 0.0,
            "prefix_rescue/suffix_kl_rows": 0.0,
            "prefix_rescue/suffix_q_retained_mass": 0.0,
            "prefix_rescue/suffix_p_retained_mass": 0.0,
            "prefix_rescue/suffix_q_residual_mass": 0.0,
            "prefix_rescue/suffix_p_residual_mass": 0.0,
        }

    eps = max(float(eps), torch.finfo(p_topk_log_probs.dtype).tiny)
    q_prob = q_logp.exp()
    q_res = q_res.to(q_prob.dtype).clamp(min=eps, max=1.0)
    p_res = p_residual_mass.to(p_topk_log_probs.dtype).clamp(min=eps, max=1.0)
    token_kl = (
        q_prob * (q_logp - p_topk_log_probs)
    ).sum(dim=-1) + q_res * (q_res.log() - p_res.log())

    if group_weight is None:
        row_weight = torch.ones(
            token_kl.shape[0], device=token_kl.device, dtype=token_kl.dtype
        )
    else:
        row_weight = group_weight.to(token_kl.dtype).view(-1).detach()
    loss = agg_loss(
        loss_mat=row_weight.unsqueeze(-1) * token_kl,
        loss_mask=mask,
        loss_agg_mode=loss_agg_mode,
    )
    valid = mask.bool()
    q_retained = q_prob.sum(dim=-1)
    p_retained = p_topk_log_probs.exp().sum(dim=-1)
    active_rows = valid.any(dim=-1)
    return loss, {
        "prefix_rescue/suffix_kl": float(loss.detach().cpu()),
        "prefix_rescue/suffix_kl_tokens": float(mask.sum().detach().cpu()),
        "prefix_rescue/suffix_kl_rows": float(active_rows.sum().detach().cpu()),
        "prefix_rescue/suffix_q_retained_mass": float(q_retained[valid].mean().detach().cpu()),
        "prefix_rescue/suffix_p_retained_mass": float(p_retained[valid].mean().detach().cpu()),
        "prefix_rescue/suffix_q_residual_mass": float(q_res[valid].mean().detach().cpu()),
        "prefix_rescue/suffix_p_residual_mass": float(p_res[valid].mean().detach().cpu()),
    }


def compute_cf_span_preference_loss(
    *,
    responses: torch.Tensor,
    log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    cf_pair_idx: torch.Tensor,
    cf_role: torch.Tensor,
    utility: torch.Tensor | None = None,
    pair_weight: torch.Tensor | None = None,
    student_entropy: torch.Tensor | None = None,
    beta_cf: float = 1.0,
    g_thresh: float = 0.0,
    preference_mode: str = "teacher_better",
    causal_probe_mode: str = "off",
    causal_probe_eta: float = 1.0,
    action_marker_token_ids: Sequence[int] | None = None,
    action_skip_token_ids: Sequence[int] | None = None,
    entropy_gate_enabled: bool = False,
    entropy_gate_threshold: float | None = None,
    entropy_gate_quantile: float = 0.75,
    entropy_gate_beta: float = 1.0,
    entropy_gate_min: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pairwise first-divergence CF span preference on explicit aux rows."""
    zero = log_probs.sum() * 0.0
    valid_pair_ids = torch.unique(cf_pair_idx[cf_pair_idx >= 0])
    if valid_pair_ids.numel() == 0:
        return zero, {
            "skill_int/cf_pairs": 0.0,
            "skill_int/cf_pairs_seen": 0.0,
            "skill_int/cf_loss": 0.0,
            "skill_int/cf_delta_mean": 0.0,
            "skill_int/cf_G_mean": 0.0,
            "skill_int/cf_g_first_mean": 0.0,
            "skill_int/cf_g_first_abs_mean": 0.0,
            "skill_int/cf_gamma_mean": 0.0,
            "skill_int/cf_gamma_min": 0.0,
            "skill_int/cf_gamma_max": 0.0,
            "skill_int/cf_action_probe_found_frac": 0.0,
            "skill_int/cf_entropy_gate_mean": 1.0,
            "skill_int/cf_group_weight_mean": 0.0,
            "skill_int/cf_abs_utility_mean": 0.0,
            "skill_int/cf_teacher_preferred_frac": 0.0,
            "skill_int/cf_student_preferred_frac": 0.0,
        }

    teacher_log_probs_d = teacher_log_probs.detach()
    mean_logp = _sample_mean(log_probs, response_mask)
    mean_skill_logp = _sample_mean(teacher_log_probs_d, response_mask)
    mean_gap = mean_skill_logp - mean_logp.detach()
    utility = utility.to(log_probs.dtype).view(-1) if utility is not None else torch.ones_like(mean_logp)
    if pair_weight is None:
        pair_weight = torch.ones_like(mean_logp)
    else:
        pair_weight = pair_weight.to(log_probs.dtype).view(-1).detach()
    entropy_gate, entropy_metrics = _entropy_exploration_gate(
        entropy=student_entropy,
        response_mask=response_mask,
        enabled=entropy_gate_enabled,
        threshold=entropy_gate_threshold,
        quantile=entropy_gate_quantile,
        beta=entropy_gate_beta,
        min_gate=entropy_gate_min,
        metric_prefix="skill_int/cf_entropy",
    )
    row_entropy_gate = _sample_mean(entropy_gate, response_mask)

    first_valid_idx, action_probe_found = _first_probe_token_indices(
        responses=responses,
        response_mask=response_mask,
        marker_token_ids=action_marker_token_ids,
        skip_token_ids=action_skip_token_ids,
    )
    row_arange = torch.arange(response_mask.shape[0], device=response_mask.device)
    g_first = teacher_log_probs_d[row_arange, first_valid_idx] - log_probs.detach()[row_arange, first_valid_idx]

    losses = []
    deltas = []
    gaps = []
    first_gaps = []
    gammas = []
    entropy_gates = []
    probe_found = []
    pair_weights = []
    utility_magnitudes = []
    preference_directions = []
    seen = 0
    for pair_id in valid_pair_ids:
        teacher_rows = torch.nonzero((cf_pair_idx == pair_id) & (cf_role == 1), as_tuple=False).view(-1)
        student_rows = torch.nonzero((cf_pair_idx == pair_id) & (cf_role == 2), as_tuple=False).view(-1)
        if teacher_rows.numel() == 0 or student_rows.numel() == 0:
            continue
        seen += 1
        t_idx = teacher_rows[0]
        s_idx = student_rows[0]
        gap = mean_gap[t_idx]
        pair_utility = utility[t_idx].detach()
        if preference_mode == "signed":
            if not bool((pair_utility != 0.0).detach().cpu()):
                continue
            direction = pair_utility.sign()
            magnitude = pair_utility.abs()
        elif preference_mode == "teacher_better":
            if not bool((pair_utility > 0.0).detach().cpu()):
                continue
            direction = torch.ones((), device=log_probs.device, dtype=log_probs.dtype)
            magnitude = torch.ones((), device=log_probs.device, dtype=log_probs.dtype)
        else:
            raise ValueError(f"Unknown CF preference mode: {preference_mode}")

        if causal_probe_mode == "one_step":
            gamma = torch.exp(-causal_probe_eta * g_first[t_idx].abs()).clamp(max=1.0)
        elif causal_probe_mode == "off":
            if preference_mode != "signed" and not bool((gap > g_thresh).detach().cpu()):
                continue
            gamma = torch.ones((), device=log_probs.device, dtype=log_probs.dtype)
        else:
            raise ValueError(f"Unknown causal_probe_mode: {causal_probe_mode}")

        delta = mean_logp[t_idx] - mean_logp[s_idx]
        pair_entropy_gate = torch.minimum(row_entropy_gate[t_idx], row_entropy_gate[s_idx]).detach()
        selected_pair_weight = pair_weight[t_idx] * magnitude
        losses.append(
            -selected_pair_weight
            * gamma
            * pair_entropy_gate
            * torch.nn.functional.logsigmoid(beta_cf * direction * delta)
        )
        deltas.append(delta.detach())
        gaps.append(gap.detach())
        first_gaps.append(g_first[t_idx].detach())
        gammas.append(gamma.detach())
        entropy_gates.append(pair_entropy_gate.detach())
        probe_found.append(action_probe_found[t_idx].detach())
        pair_weights.append(selected_pair_weight.detach())
        utility_magnitudes.append(magnitude.detach())
        preference_directions.append(direction.detach())

    if not losses:
        metrics = {
            "skill_int/cf_pairs": 0.0,
            "skill_int/cf_pairs_seen": float(seen),
            "skill_int/cf_loss": 0.0,
            "skill_int/cf_delta_mean": 0.0,
            "skill_int/cf_G_mean": 0.0,
            "skill_int/cf_g_first_mean": 0.0,
            "skill_int/cf_g_first_abs_mean": 0.0,
            "skill_int/cf_gamma_mean": 0.0,
            "skill_int/cf_gamma_min": 0.0,
            "skill_int/cf_gamma_max": 0.0,
            "skill_int/cf_action_probe_found_frac": 0.0,
            "skill_int/cf_entropy_gate_mean": 1.0,
            "skill_int/cf_group_weight_mean": 0.0,
            "skill_int/cf_abs_utility_mean": 0.0,
            "skill_int/cf_teacher_preferred_frac": 0.0,
            "skill_int/cf_student_preferred_frac": 0.0,
        }
        metrics.update(entropy_metrics)
        return zero, metrics

    loss = torch.stack(losses).mean()
    delta_t = torch.stack(deltas)
    gap_t = torch.stack(gaps)
    first_gap_t = torch.stack(first_gaps)
    gamma_t = torch.stack(gammas)
    entropy_gate_t = torch.stack(entropy_gates)
    probe_found_t = torch.stack(probe_found).to(torch.float32)
    pair_weight_t = torch.stack(pair_weights)
    utility_magnitude_t = torch.stack(utility_magnitudes)
    preference_direction_t = torch.stack(preference_directions)
    metrics = {
        "skill_int/cf_pairs": float(len(losses)),
        "skill_int/cf_pairs_seen": float(seen),
        "skill_int/cf_loss": float(loss.detach().cpu()),
        "skill_int/cf_delta_mean": float(delta_t.mean().cpu()),
        "skill_int/cf_G_mean": float(gap_t.mean().cpu()),
        "skill_int/cf_g_first_mean": float(first_gap_t.mean().cpu()),
        "skill_int/cf_g_first_abs_mean": float(first_gap_t.abs().mean().cpu()),
        "skill_int/cf_gamma_mean": float(gamma_t.mean().cpu()),
        "skill_int/cf_gamma_min": float(gamma_t.min().cpu()),
        "skill_int/cf_gamma_max": float(gamma_t.max().cpu()),
        "skill_int/cf_action_probe_found_frac": float(probe_found_t.mean().cpu()),
        "skill_int/cf_entropy_gate_mean": float(entropy_gate_t.mean().cpu()),
        "skill_int/cf_group_weight_mean": float(pair_weight_t.mean().cpu()),
        "skill_int/cf_abs_utility_mean": float(utility_magnitude_t.mean().cpu()),
        "skill_int/cf_teacher_preferred_frac": float((preference_direction_t > 0).to(torch.float32).mean().cpu()),
        "skill_int/cf_student_preferred_frac": float((preference_direction_t < 0).to(torch.float32).mean().cpu()),
    }
    metrics.update(entropy_metrics)
    return loss, metrics
