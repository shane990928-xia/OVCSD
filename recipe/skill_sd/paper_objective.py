"""Per-token coefficients for the three independently normalized paper terms."""
from collections import Counter

import torch


def prepare_weights(proto, action_mask_fn):
    batch = proto.batch
    n = len(proto)
    mask = action_mask_fn(proto).bool()
    aux = batch.get("skill_int_is_teacher_aux", torch.zeros(n)).bool()
    local = batch.get("skill_int_is_rescue_pg", torch.zeros(n)).bool()
    suffix = batch.get("skill_int_is_rescue_suffix", torch.zeros(n)).bool()
    positive = local & (batch["advantages"].amax(dim=1) > 0)
    student = ~(aux | positive)
    tasks = list(map(str, proto.non_tensor_batch["uid"]))
    trajectories = list(map(str, proto.non_tensor_batch["traj_uid"]))
    steps = proto.non_tensor_batch["step_idx"]
    counts = mask.sum(dim=1).tolist()

    # adjust_batch can repeat final rows to fill a distributed shard.
    row_keys = list(zip(tasks, trajectories, map(str, steps)))
    replicas = Counter(row_keys[i] for i in range(n) if student[i])
    tokens = Counter()
    task_trajectories = {}
    for i, (task, traj, step) in enumerate(row_keys):
        if student[i]:
            tokens[task, traj] += counts[i] / replicas[row_keys[i]]
            task_trajectories.setdefault(task, set()).add(traj)
    sites = int(positive.sum())
    suffix_tokens = Counter()
    for i in range(n):
        if suffix[i]:
            suffix_tokens[tasks[i]] += counts[i]
    base_weight = torch.zeros(n)
    pg_weight = torch.zeros(n)
    suffix_weight = torch.zeros(n)
    for i, (task, traj, step) in enumerate(row_keys):
        if student[i] and tokens[task, traj] > 0:
            base_weight[i] = 1 / (len(task_trajectories) * len(task_trajectories[task])
                                  * tokens[task, traj] * replicas[row_keys[i]])
            pg_weight[i] = base_weight[i]
        if local[i] and counts[i]:
            magnitude = batch["advantages"][i].abs().max().item()
            members = round(magnitude**2 if positive[i] else magnitude**-2)
            pg_weight[i] = 1 / (sites * (members + 1) * counts[i])
        if suffix[i] and counts[i]:
            suffix_weight[i] = 1 / (len(suffix_tokens) * suffix_tokens[task])
    # The actor scales a sampled PPO minibatch by total rows / minibatch rows.
    batch["paper_mask"] = mask
    batch["paper_pg_weight"] = n * pg_weight
    batch["paper_ref_weight"] = n * base_weight
    batch["paper_suffix_weight"] = n * suffix_weight
