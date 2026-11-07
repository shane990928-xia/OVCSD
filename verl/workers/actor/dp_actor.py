# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import gzip
import json
import time
import logging
import os
from typing import Tuple

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _compute_cf_loss_once(
    *,
    data,
    responses,
    log_prob,
    response_mask,
    entropy,
    lambda_cf: float,
    beta_cf: float,
    g_thresh: float,
    preference_mode: str = "teacher_better",
    causal_probe_mode: str = "off",
    causal_probe_eta: float = 1.0,
    action_marker_token_ids=None,
    action_skip_token_ids=None,
    entropy_gate_enabled: bool = False,
    entropy_gate_threshold=None,
    entropy_gate_quantile: float = 0.75,
    entropy_gate_beta: float = 1.0,
    entropy_gate_min: float = 0.0,
):
    if lambda_cf <= 0 or "skill_int_is_teacher_aux" not in data:
        zero = log_prob.sum() * 0.0
        return zero, {}
    teacher_aux_rows = data["skill_int_is_teacher_aux"].bool().view(-1)
    if not teacher_aux_rows.any():
        zero = log_prob.sum() * 0.0
        return zero, {}

    from recipe.skill_sd.teacher_aux_loss import compute_cf_span_preference_loss

    cf_loss, cf_metrics = compute_cf_span_preference_loss(
        responses=responses,
        log_probs=log_prob,
        teacher_log_probs=data["skill_int_teacher_log_probs"],
        response_mask=response_mask,
        cf_pair_idx=data["skill_int_cf_pair_idx"].long().view(-1),
        cf_role=data["skill_int_cf_role"].long().view(-1),
        utility=data["skill_int_utility"],
        pair_weight=data.get("skill_int_group_weight"),
        student_entropy=entropy,
        beta_cf=beta_cf,
        g_thresh=g_thresh,
        preference_mode=preference_mode,
        causal_probe_mode=causal_probe_mode,
        causal_probe_eta=causal_probe_eta,
        action_marker_token_ids=action_marker_token_ids,
        action_skip_token_ids=action_skip_token_ids,
        entropy_gate_enabled=entropy_gate_enabled,
        entropy_gate_threshold=entropy_gate_threshold,
        entropy_gate_quantile=entropy_gate_quantile,
        entropy_gate_beta=entropy_gate_beta,
        entropy_gate_min=entropy_gate_min,
    )
    cf_metrics["skill_int/effective_lambda_cf"] = lambda_cf
    cf_metrics["skill_int/causal_probe_mode_one_step"] = float(causal_probe_mode == "one_step")
    return lambda_cf * cf_loss, cf_metrics


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_suffix_distribution(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
        temperature: float,
        top_k: int = None,
        gather_token_ids: torch.Tensor = None,
    ):
        """Return top-k probabilities without materializing a full softmax."""
        if (top_k is None) == (gather_token_ids is None):
            raise ValueError("exactly one of top_k or gather_token_ids is required")
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            if not hasattr(output, "logits"):
                raise RuntimeError(
                    "suffix KL requires actor logits; disable fused-kernel logprob-only output"
                )
            logits = output.logits[:, -int(response_length) - 1 : -1, :]
            logits = logits / max(float(temperature), 1.0e-8)
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            if gather_token_ids is None:
                k = min(max(int(top_k), 1), int(logits.shape[-1]))
                selected_logits, token_ids = torch.topk(logits, k=k, dim=-1)
            else:
                token_ids = gather_token_ids.to(device=logits.device, dtype=torch.long)
                selected_logits = torch.gather(logits, dim=-1, index=token_ids)
            selected_log_probs = selected_logits - log_z
            retained_mass = selected_log_probs.exp().sum(dim=-1)
            residual_mass = (1.0 - retained_mass).clamp(min=0.0, max=1.0)
            return selected_log_probs, residual_mass, token_ids

    def _precompute_rescue_suffix_q_cache(
        self,
        batch,
        *,
        top_k: int,
        temperature: float,
    ):
        """Freeze theta-old skill distributions before any optimizer step."""
        suffix_rows = batch["skill_int_is_rescue_suffix"].bool().view(-1)
        row_ids = torch.nonzero(suffix_rows, as_tuple=False).view(-1)
        if not row_ids.numel():
            return None
        response_length = int(batch["responses"].shape[-1])
        micro_batch_size = max(
            int(self.config.ppo_micro_batch_size_per_gpu), 1
        )
        try:
            device = next(self.actor_module.parameters()).device
        except StopIteration:
            device = get_torch_device().current_device()
        cached_logp = []
        cached_residual = []
        cached_ids = []
        with torch.no_grad():
            for row_chunk in row_ids.split(micro_batch_size):
                q_logp, q_residual, q_ids = self._forward_suffix_distribution(
                    input_ids=batch["skill_int_suffix_q_input_ids"][row_chunk].to(device),
                    attention_mask=batch["skill_int_suffix_q_attention_mask"][row_chunk].to(device),
                    position_ids=batch["skill_int_suffix_q_position_ids"][row_chunk].to(device),
                    response_length=response_length,
                    temperature=temperature,
                    top_k=top_k,
                )
                cached_logp.append(q_logp.detach().to("cpu", dtype=torch.float32))
                cached_residual.append(
                    q_residual.detach().to("cpu", dtype=torch.float32)
                )
                cached_ids.append(q_ids.detach().to("cpu", dtype=torch.long))
        q_logp = torch.cat(cached_logp, dim=0)
        q_residual = torch.cat(cached_residual, dim=0)
        q_ids = torch.cat(cached_ids, dim=0)
        n_rows = int(batch["responses"].shape[0])
        k = int(q_ids.shape[-1])
        full_logp = torch.zeros(
            n_rows, response_length, k, dtype=torch.float32
        )
        full_residual = torch.zeros(
            n_rows, response_length, dtype=torch.float32
        )
        full_ids = torch.zeros(
            n_rows, response_length, k, dtype=torch.long
        )
        cpu_rows = row_ids.detach().to("cpu", dtype=torch.long)
        full_logp[cpu_rows] = q_logp
        full_residual[cpu_rows] = q_residual
        full_ids[cpu_rows] = q_ids
        return {
            "skill_int_suffix_q_topk_log_probs": full_logp,
            "skill_int_suffix_q_residual_mass": full_residual,
            "skill_int_suffix_q_topk_ids": full_ids,
        }

    def _archive_rescue_suffix_q_cache(self, batch, cache, meta_info) -> None:
        """Write exact frozen q distributions to a rank-sharded gzip archive."""
        archive_dir = meta_info.get("skill_int_suffix_q_archive_dir")
        run_id = meta_info.get("skill_int_suffix_q_archive_run_id")
        if not archive_dir or not run_id or cache is None:
            return
        os.makedirs(str(archive_dir), exist_ok=True)
        rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        step = int(meta_info.get("skill_int_suffix_q_archive_step", 0))
        path = os.path.join(
            str(archive_dir),
            f"q_cache_{run_id}_step{step:06d}_rank{rank}.jsonl.gz",
        )
        suffix_rows = torch.nonzero(
            batch["skill_int_is_rescue_suffix"].bool().view(-1),
            as_tuple=False,
        ).view(-1)
        action_mask = batch["skill_int_suffix_action_mask"].bool()
        responses = batch["responses"]
        node_hashes = batch["skill_int_suffix_archive_node_hash"].view(-1)
        env_steps = batch["skill_int_suffix_archive_step"].view(-1)
        q_ids = cache["skill_int_suffix_q_topk_ids"]
        q_logp = cache["skill_int_suffix_q_topk_log_probs"]
        q_residual = cache["skill_int_suffix_q_residual_mass"]
        with gzip.open(path, "at", encoding="utf-8") as fh:
            for row in suffix_rows.detach().cpu().tolist():
                for token_pos in torch.nonzero(
                    action_mask[row], as_tuple=False
                ).view(-1).detach().cpu().tolist():
                    record = {
                        "schema_version": "cpsd.rescue.qcache.v1",
                        "run_instance_id": str(run_id),
                        "global_step": step,
                        "rank": rank,
                        "worker_row": int(row),
                        "node_hash": int(node_hashes[row].detach().cpu()),
                        "environment_step": int(env_steps[row].detach().cpu()),
                        "token_position": int(token_pos),
                        "target_token_id": int(
                            responses[row, token_pos].detach().cpu()
                        ),
                        "q_topk_ids": q_ids[row, token_pos].tolist(),
                        "q_topk_log_probs": q_logp[row, token_pos].tolist(),
                        "q_residual_mass": float(q_residual[row, token_pos]),
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Optimize clipped skill-free GRPO, verified local PG and frozen-teacher KL."""
        from recipe.skill_sd.teacher_aux_loss import compute_rescue_suffix_topk_kl

        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        # Ray may lock the transported TensorDict. The cache belongs to this
        # update only, so keep a mutable local container sharing its tensors.
        batch = TensorDict(dict(data.batch.items()), batch_size=data.batch.batch_size,
                           device=data.batch.device)
        suffix_key = "skill_int_is_rescue_suffix"
        if suffix_key in batch and batch[suffix_key].any():
            cache = self._precompute_rescue_suffix_q_cache(batch, top_k=64, temperature=1.0)
            for key, tensor in cache.items():
                batch[key] = tensor.to(batch["responses"].device)
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini in batch.split(self.config.ppo_mini_batch_size):
                self.actor_optimizer.zero_grad()
                for micro in mini.split(self.config.ppo_micro_batch_size_per_gpu):
                    mask = micro["paper_mask"].float()
                    _, logp = self._forward_micro_batch(micro, temperature, calculate_entropy=False)
                    ratio = (logp - micro["old_log_probs"]).clamp(-20, 20).exp()
                    adv = micro["advantages"]
                    clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    surrogate = torch.minimum(ratio * adv, clipped * adv)
                    pg = -(surrogate * mask * micro["paper_pg_weight"][:, None]).sum()
                    kld = kl_penalty(logp, micro["ref_log_prob"], self.config.kl_loss_type)
                    ref = (kld * mask * micro["paper_ref_weight"][:, None]).sum()
                    loss = pg + self.config.kl_loss_coef * ref
                    if suffix_key in micro and micro[suffix_key].any():
                        rows = micro[suffix_key].bool()
                        pred, residual, _ = self._forward_suffix_distribution(
                            input_ids=micro["input_ids"][rows], attention_mask=micro["attention_mask"][rows],
                            position_ids=micro["position_ids"][rows], response_length=mask.shape[-1],
                            temperature=1.0, gather_token_ids=micro["skill_int_suffix_q_topk_ids"][rows],
                        )
                        suffix_loss, _ = compute_rescue_suffix_topk_kl(
                            q_topk_log_probs=micro["skill_int_suffix_q_topk_log_probs"][rows],
                            q_residual_mass=micro["skill_int_suffix_q_residual_mass"][rows],
                            p_topk_log_probs=pred, p_residual_mass=residual, response_mask=mask[rows],
                            group_weight=micro["paper_suffix_weight"][rows], loss_agg_mode="token-mean",
                        )
                        suffix_total = suffix_loss * mask[rows].sum()
                        loss = loss + 0.05 * suffix_total
                        append_to_dict(metrics, {"actor/suffix_kl": suffix_total.detach().item()})
                    (loss / len(mini)).backward()
                    append_to_dict(metrics, {"actor/pg_loss": pg.detach().item() / len(micro),
                                             "actor/kl_loss": ref.detach().item() / len(micro)})
                norm = self._optimizer_step()
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite actor gradient")
                append_to_dict(metrics, {"actor/grad_norm": norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics
