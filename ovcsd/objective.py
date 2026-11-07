"""Supplement Eqs. 7 and 17–29. Each backward call releases one action graph."""
import math
from collections import Counter

import torch
from torch.nn import functional as F

from .method import advantages


def clipped(logp, old, advantage):
    ratio = (logp - old.detach()).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)


def coarsened_kl(teacher, student, k=64):
    """Exact forward KL on the teacher's top k and a single residual symbol."""
    q = teacher.detach().float().log_softmax(-1)
    p = student.float().log_softmax(-1)
    k = min(k, q.shape[-1])
    q_top, ids = q.topk(k, dim=-1)
    p_top = p.gather(-1, ids)
    loss = (q_top.exp() * (q_top - p_top)).sum(-1)
    if k < q.shape[-1]:
        # Sum the true tail in log space, including when top-k mass rounds to one.
        q_tail = q.scatter(-1, ids, -torch.inf).logsumexp(-1)
        p_tail = p.scatter(-1, ids, -torch.inf).logsumexp(-1)
        loss = loss + q_tail.exp() * (q_tail - p_tail)
    return loss


def log_probs(policy, episode, depth, model=None):
    action = episode.steps[depth].action
    logits = policy.logits(episode, depth, model=model)
    targets = torch.tensor([action.response_ids[i] for i in action.positions], device=logits.device)
    return -F.cross_entropy(logits, targets, reduction="none")


def accumulate(policy, reference, groups, sites):
    """One epoch and one optimizer step per batch. Parameters are fixed until
    every term has been scored. Detaching the skill-free forward thus gives the
    exact rollout snapshot denominator, also for skill-generated actions.
    """
    stats = Counter()

    def backward(term, name):
        stats[name] += term.detach().item()
        term.backward()

    for group in groups:
        adv, _ = advantages(group)
        for episode, a in zip(group, adv):
            n = sum(len(step.action.positions) for step in episode.steps)
            denom = len(groups) * len(group) * n
            for d, step in enumerate(episode.steps):
                if not step.action.positions:
                    continue
                with torch.no_grad():
                    ref = log_probs(policy, episode, d, reference)
                current = log_probs(policy, episode, d)
                base = clipped(current, current.detach(), a).sum() / denom
                delta = ref - current
                ref_kl = (delta.expm1() - delta).sum() / denom
                stats["base"] += base.detach().item()
                stats["ref_kl"] += ref_kl.detach().item()
                backward(base + .01 * ref_kl, "regularized_base")

    suffix_counts = [sum(len(s.action.positions) for s in site.suffix) for site in sites]
    nonempty = sum(n > 0 for n in suffix_counts)
    for site, n_suffix in zip(sites, suffix_counts):
        m = len(site.members)
        rows = [(site.teacher, math.sqrt(m))] + [(student, -1 / math.sqrt(m)) for student in site.members]
        for episode, a in rows:
            current = log_probs(policy, episode, site.depth)
            local = clipped(current, current.detach(), a).mean()
            backward(local / ((m + 1) * len(sites)), "divergence")
        for d in range(site.depth + 1, len(site.teacher.steps)):
            if not site.teacher.steps[d].action.positions:
                continue
            with torch.no_grad():
                privileged_logits = policy.logits(site.teacher, d, site.skills)
            student_logits = policy.logits(site.teacher, d)
            term = coarsened_kl(privileged_logits, student_logits).sum() / (n_suffix * nonempty)
            backward(.05 * term, "weighted_suffix")
    return stats
