"""Action parsing, prompt construction and replay helpers for environment branches."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from recipe.skill_sd.phi_compiler import Milestone, delta_phi, phi_curve, phi_value


_ADMISSIBLE_SECTION_RE = re.compile(
    r"Your admissible actions[^:]*:\s*\[(?P<body>.*?)\]\."
    r"(?=\s*\n\s*Now it's your turn|\s*$)",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_ACTION_RE = re.compile(r"['\"](?P<action>[^'\"]+)['\"]")


def extract_alfworld_admissible_actions(prompt: str) -> tuple[str, ...]:
    """Extract the quoted admissible-action list from an ALFWorld prompt."""

    match = _ADMISSIBLE_SECTION_RE.search(str(prompt or ""))
    if match is None:
        return ()
    return tuple(
        item.group("action").strip()
        for item in _QUOTED_ACTION_RE.finditer(match.group("body"))
        if item.group("action").strip()
    )


@dataclass(frozen=True)
class CFBranchPair:
    """One counterfactual branch pair at a Φ-stall divergence point."""

    pair_id: str
    group_uid: str
    session: str
    student_traj_uid: str
    student_row: int          # the student batch row at the stall point h_τ
    stall_step: int           # step index of h_τ within the trajectory
    teacher_action: str       # a_τ^T (with-skill teacher's action at h_τ)
    student_action: str       # a_τ^S (student's actual action at h_τ)
    prefix_actions: tuple     # student actions for steps 1..h_τ-1 (shared prefix)
    n_continuation: int       # min(n, remaining) actually run
    phi_teacher_end: float    # Φ at teacher-branch end
    phi_student_end: float    # Φ at student-branch end
    a_cf: float               # signed A^cf (utility)
    ran_to_end: bool = False  # whether adaptive granularity ran the branch to done


@dataclass(frozen=True)
class StallCandidate:
    """A candidate stall point selected for potential branching."""

    student_row: int
    stall_step: int
    delta_phi: float          # ΔΦ at this step (≤ 0)
    student_action: str


def select_phi_stall_points(
    *,
    env: str,
    milestones: Sequence[Milestone],
    observations: Sequence[str],
    actions: Sequence[str],
    row_indices: Sequence[int],
    top_b: int,
    gamma: float = 0.95,
    terminal_reward: Optional[float] = None,
) -> List[StallCandidate]:
    """Return the top-B ΔΦ≤0 stall points on one student trajectory.

    ``observations``/``actions`` are the per-step lists for the trajectory
    (ordered by step). ``row_indices[t]`` is the batch row of step t. Stall =
    ΔΦ_t ≤ 0 (Φ did not advance from step t to t+1). We branch at step t (the
    action that failed to advance), so the candidate's row is row_indices[t].
    """
    n = min(len(observations), len(actions), len(row_indices))
    if n < 2:
        return []
    curve = phi_curve(env, milestones, observations[:n], actions[:n], terminal_reward)
    dphi = delta_phi(curve, gamma=gamma)  # length n-1, index t = step t -> t+1
    candidates = [
        StallCandidate(
            student_row=int(row_indices[t]),
            stall_step=t,
            delta_phi=float(dphi[t]),
            student_action=str(actions[t]),
        )
        for t in range(len(dphi))
        if dphi[t] <= 1e-9
    ]
    # Most-negative ΔΦ first (most stuck); tie-break by earliest step.
    candidates.sort(key=lambda c: (c.delta_phi, c.stall_step))
    if top_b is not None and top_b >= 0:
        candidates = candidates[:top_b]
    return candidates


def compute_a_cf(
    *,
    phi_teacher_end: float,
    phi_student_end: float,
    gamma: float,
    n: int,
    reduce: str = "gamma_n",
    phi_teacher_curve: Optional[Sequence[float]] = None,
    phi_student_curve: Optional[Sequence[float]] = None,
) -> float:
    """A^cf between the two branches.

    - ``gamma_n``: γ^n[Φ(end^T) − Φ(end^S)] (endpoint difference).
    - ``max_k``:   max_{1≤k≤n} γ^k[Φ_k^T − Φ_k^S] (peak advantage over the branch;
      needs the per-step branch Φ curves).
    """
    if reduce == "gamma_n":
        return (gamma ** max(int(n), 1)) * (float(phi_teacher_end) - float(phi_student_end))
    if reduce == "max_k":
        if not phi_teacher_curve or not phi_student_curve:
            # Fall back to endpoint form if curves not provided.
            return (gamma ** max(int(n), 1)) * (float(phi_teacher_end) - float(phi_student_end))
        k_max = min(len(phi_teacher_curve), len(phi_student_curve))
        return max(
            (gamma ** (k + 1)) * (float(phi_teacher_curve[k]) - float(phi_student_curve[k]))
            for k in range(k_max)
        )
    raise ValueError(f"Unknown a_cf reduce mode: {reduce}")


def build_branch_pairs(
    *,
    branch_results: Sequence[dict],
    gamma: float,
    epsilon_cf: float = 0.0,
    a_cf_reduce: str = "gamma_n",
) -> List[CFBranchPair]:
    """Assemble CFBranchPair list from executed branch results.

    Each ``branch_results`` item is a dict produced by the env driver with keys:
      group_uid, session, student_traj_uid, student_row, stall_step,
      teacher_action, student_action, prefix_actions, n_continuation,
      phi_teacher_end, phi_student_end, ran_to_end,
      (optional) phi_teacher_curve, phi_student_curve.

    Confirmed-divergence filtering (a_τ^T ≠ a_τ^S) is done by the caller before
    branching; here we apply the ε_cf boundary filter: |A^cf| ≤ ε_cf -> drop
    (no preference). The sign of A^cf sets the preference
    direction (signed mode).
    """
    pairs: List[CFBranchPair] = []
    for r in branch_results:
        a_cf = compute_a_cf(
            phi_teacher_end=r["phi_teacher_end"],
            phi_student_end=r["phi_student_end"],
            gamma=gamma,
            n=r.get("n_continuation", 1),
            reduce=a_cf_reduce,
            phi_teacher_curve=r.get("phi_teacher_curve"),
            phi_student_curve=r.get("phi_student_curve"),
        )
        if abs(a_cf) <= float(epsilon_cf):
            continue  # boundary filter: no preference
        pairs.append(
            CFBranchPair(
                pair_id=f"{r['group_uid']}:{r['student_traj_uid']}:{r['stall_step']}",
                group_uid=str(r["group_uid"]),
                session=str(r["session"]),
                student_traj_uid=str(r["student_traj_uid"]),
                student_row=int(r["student_row"]),
                stall_step=int(r["stall_step"]),
                teacher_action=str(r["teacher_action"]),
                student_action=str(r["student_action"]),
                prefix_actions=tuple(r.get("prefix_actions", ())),
                n_continuation=int(r.get("n_continuation", 1)),
                phi_teacher_end=float(r["phi_teacher_end"]),
                phi_student_end=float(r["phi_student_end"]),
                a_cf=float(a_cf),
                ran_to_end=bool(r.get("ran_to_end", False)),
            )
        )
    return pairs


def summarize_branch_pairs(pairs: Sequence[CFBranchPair]) -> dict:
    """Metrics for wandb."""
    if not pairs:
        return {
            "cf_branch/pairs_emitted": 0.0,
            "cf_branch/a_cf_mean": 0.0,
            "cf_branch/teacher_preferred_frac": 0.0,
            "cf_branch/student_preferred_frac": 0.0,
            "cf_branch/run_to_end_frac": 0.0,
        }
    n = len(pairs)
    teacher_pref = sum(1 for p in pairs if p.a_cf > 0)
    student_pref = sum(1 for p in pairs if p.a_cf < 0)
    ran_end = sum(1 for p in pairs if p.ran_to_end)
    return {
        "cf_branch/pairs_emitted": float(n),
        "cf_branch/a_cf_mean": float(sum(p.a_cf for p in pairs) / n),
        "cf_branch/teacher_preferred_frac": float(teacher_pref / n),
        "cf_branch/student_preferred_frac": float(student_pref / n),
        "cf_branch/run_to_end_frac": float(ran_end / n),
    }


# --------------------------------------------------------------------------- #
# Env-driver pure helpers (prompt construction, prefix extraction, record
# assembly) — the Ray/model-touching orchestration lives in the trainer, these
# are the testable pieces.
# --------------------------------------------------------------------------- #
WEBSHOP_NO_SKILL_TEMPLATE = (
    "\nYou are an expert autonomous agent operating in the WebShop e-commerce environment. \n"
    "Your task is to: {task_description}.\n"
    "Your current observation is: {current_observation}.\n"
    "Your admissible actions of the current situation are: \n[\n{available_actions}\n].\n\n"
    "Now it's your turn to take one action for the current step.\n"
    "You should first reason step-by-step about the current situation, then think carefully "
    "which admissible action best advances the shopping goal. This reasoning process MUST be "
    "enclosed within <think> </think> tags. \n"
    "Once you've finished your reasoning, you should choose an admissible action for current "
    "step and present it within <action> </action> tags.\n"
)

SEARCHQA_NO_SKILL_TEMPLATE = (
    "\nYou are an expert agent tasked with answering the given question step-by-step.\n"
    "Your question: {task_description}\n\n"
    "Prior interaction history:\n{current_observation}\n\n"
    "Now it's your turn to respond for the current step.\n"
    "You should first conduct reasoning process. This process MUST be enclosed within "
    "<think> </think> tags.\n"
    "After completing your reasoning, choose only one action: use <search> query </search> "
    "for more evidence, or provide the final answer in <answer> </answer> tags.\n"
)

SEARCHQA_WITH_SKILL_TEMPLATE = (
    "\nYou are an expert agent tasked with answering the given question step-by-step.\n"
    "Your question: {task_description}\n\n"
    "## Retrieved Relevant Experience\n\n{retrieved_memories}\n\n"
    "Prior interaction history:\n{current_observation}\n\n"
    "Now it's your turn to respond for the current step.\n"
    "You should first conduct reasoning process. This process MUST be enclosed within "
    "<think> </think> tags.\n"
    "After completing your reasoning, choose only one action: use <search> query </search> "
    "for more evidence, or provide the final answer in <answer> </answer> tags.\n"
)


def format_available_actions(available_actions: Sequence[str]) -> str:
    """Mirror env_manager.format_avail_actions rendering (one quoted action per line)."""
    return "\n".join(f"'{s}'," for s in available_actions)


def build_no_skill_prompt(task_description: str, observation: str,
                          available_actions: Sequence[str], env: str = "webshop", *,
                          step_count: int = 0, history_length: int = 0,
                          action_history: str = "") -> str:
    """Reconstruct the NO-SKILL prompt from raw env obs (branch continuation uses the
    student's no-skill policy). ALFWorld continuations use the same task/history
    fields as the normal environment manager once at least one history item exists."""
    if env == "alfworld":
        if int(step_count) > 0 and int(history_length) > 0:
            return ALFWORLD_HISTORY_NO_SKILL_TEMPLATE.format(
                task_description=task_description,
                step_count=int(step_count),
                history_length=int(history_length),
                action_history=action_history,
                current_step=int(step_count) + 1,
                current_observation=observation,
                admissible_actions=format_admissible_actions(available_actions),
            )
        return ALFWORLD_NO_SKILL_TEMPLATE.format(
            current_observation=observation,
            admissible_actions=format_admissible_actions(available_actions),
        )
    if env == "searchqa":
        return SEARCHQA_NO_SKILL_TEMPLATE.format(
            task_description=task_description,
            current_observation=observation,
        )
    return WEBSHOP_NO_SKILL_TEMPLATE.format(
        task_description=task_description,
        current_observation=observation,
        available_actions=format_available_actions(available_actions),
    )


# WITH-SKILL continuation prompt. Mirrors WEBSHOP_TEMPLATE_NO_HIS_WITH_MEMORY
# (agent_system/environments/prompts/webshop.py) so the teacher branch continues
# under the SAME model + skill context (OPSD: teacher = model+skills, student =
# model, no skills). The {retrieved_memories} block is the task-level skill text.
WEBSHOP_WITH_SKILL_TEMPLATE = (
    "\nYou are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
    "Your task is to: {task_description}.\n\n"
    "## Retrieved Relevant Experience\n\n"
    "{retrieved_memories}\n\n"
    "Your current observation is: {current_observation}.\n"
    "Your admissible actions of the current situation are:\n[\n{available_actions}\n].\n\n"
    "Now it's your turn to take one action for the current step.\n"
    "You should first reason step-by-step about the current situation, then think carefully "
    "which admissible action best advances the shopping goal. This reasoning process MUST be "
    "enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for current "
    "step and present it within <action> </action> tags.\n"
)

# --------------------------------------------------------------------------- #
# ALFWorld continuation prompts. The no-history forms mirror the initial prompt;
# the history forms mirror ALFWORLD_TEMPLATE[_WITH_MEMORY], including the explicit
# task and recent observation/action history required after the initial state.
# Admissible actions are one quoted action per line, excluding 'help'.
# --------------------------------------------------------------------------- #
ALFWORLD_NO_SKILL_TEMPLATE = (
    "\nYou are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning "
    "process MUST be enclosed within <think> </think> tags. \n"
    "Once you've finished your reasoning, you should choose an admissible action for "
    "current step and present it within <action> </action> tags.\n"
)

ALFWORLD_WITH_SKILL_TEMPLATE = (
    "\nYou are an expert agent operating in the ALFRED Embodied Environment.\n\n"
    "## Retrieved Relevant Experience\n\n"
    "{retrieved_memories}\n\n"
    "Your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning "
    "process MUST be enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for "
    "current step and present it within <action> </action> tags.\n"
)

ALFWORLD_HISTORY_NO_SKILL_TEMPLATE = (
    "\nYou are an expert agent operating in the ALFRED Embodied Environment. "
    "Your task is to: {task_description}\n"
    "Prior to this step, you have already taken {step_count} step(s). Below are the "
    "most recent {history_length} observations and the corresponding actions you took: "
    "{action_history}\n"
    "You are now at step {current_step} and your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning "
    "process MUST be enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for "
    "current step and present it within <action> </action> tags.\n"
)

ALFWORLD_HISTORY_WITH_SKILL_TEMPLATE = (
    "\nYou are an expert agent operating in the ALFRED Embodied Environment. "
    "Your task is to: {task_description}\n\n"
    "## Retrieved Relevant Experience\n\n"
    "{retrieved_memories}\n\n"
    "## Current Progress\n\n"
    "Prior to this step, you have already taken {step_count} step(s). Below are the "
    "most recent {history_length} observations and the corresponding actions you took: "
    "{action_history}\n"
    "You are now at step {current_step} and your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning "
    "process MUST be enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for "
    "current step and present it within <action> </action> tags.\n"
)


def format_admissible_actions(available_actions: Sequence[str]) -> str:
    """Mirror env_manager.build_text_obs ALFWorld rendering (one quoted action per
    line joined by '\\n ', excluding 'help')."""
    return "\n ".join(f"'{s}'" for s in available_actions if s != "help")


# Header marking the skill block inside a with-skill (alt) prompt.
_SKILL_SECTION_HEADER = "## Retrieved Relevant Experience"


def build_with_skill_prompt(task_description: str, observation: str,
                            available_actions: Sequence[str], skill_text: str,
                            env: str = "webshop", *, step_count: int = 0,
                            history_length: int = 0, action_history: str = "") -> str:
    """Reconstruct the WITH-SKILL prompt for teacher-branch continuation. ``skill_text``
    is the task-level retrieved-experience block (static across a task's steps). Falls
    back to the no-skill prompt if skill_text is empty. ``env`` selects the template."""
    if not skill_text or not str(skill_text).strip():
        prompt = build_no_skill_prompt(
            task_description,
            observation,
            available_actions,
            env=env,
            step_count=step_count,
            history_length=history_length,
            action_history=action_history,
        )
        return prompt
    if env == "alfworld":
        if int(step_count) > 0 and int(history_length) > 0:
            prompt = ALFWORLD_HISTORY_WITH_SKILL_TEMPLATE.format(
                task_description=task_description,
                retrieved_memories=str(skill_text).strip(),
                step_count=int(step_count),
                history_length=int(history_length),
                action_history=action_history,
                current_step=int(step_count) + 1,
                current_observation=observation,
                admissible_actions=format_admissible_actions(available_actions),
            )
        else:
            prompt = ALFWORLD_WITH_SKILL_TEMPLATE.format(
                retrieved_memories=str(skill_text).strip(),
                current_observation=observation,
                admissible_actions=format_admissible_actions(available_actions),
            )
        return prompt
    if env == "searchqa":
        return SEARCHQA_WITH_SKILL_TEMPLATE.format(
            task_description=task_description,
            retrieved_memories=str(skill_text).strip(),
            current_observation=observation,
        )
    return WEBSHOP_WITH_SKILL_TEMPLATE.format(
        task_description=task_description,
        retrieved_memories=str(skill_text).strip(),
        current_observation=observation,
        available_actions=format_available_actions(available_actions),
    )


def extract_skill_text(alt_prompt_text: Optional[str]) -> str:
    """Pull the task-level skill block from a with-skill (alt) prompt.

    The alt prompt view carries the retrieved experience under a
    '## Retrieved Relevant Experience' header (see WEBSHOP_*_WITH_MEMORY). We slice
    from that header up to the 'Your current observation' / admissible-actions
    section (the per-step, non-static part). Returns '' if not found (teacher
    branch then falls back to no-skill continuation)."""
    if not alt_prompt_text:
        return ""
    text = str(alt_prompt_text)
    start = text.find(_SKILL_SECTION_HEADER)
    if start < 0:
        return ""
    body = text[start + len(_SKILL_SECTION_HEADER):]
    # Cut at the first per-step marker that follows the skill block.
    cuts = [body.find(m) for m in ("## Current Progress", "Your current observation",
                                   "Prior to this step", "Your admissible actions")]
    cuts = [c for c in cuts if c >= 0]
    if cuts:
        body = body[:min(cuts)]
    return body.strip()


def _normalize_alfworld_action(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split()).strip(" .,;:\t\r\n")


def _project_alfworld_action(
    model_response: str, available_actions: Sequence[str]
) -> Optional[str]:
    """Project a noisy response onto one exact action exposed by ALFWorld."""

    lookup = {
        normalized: normalized
        for action in available_actions
        if (normalized := _normalize_alfworld_action(action))
    }
    if not lookup:
        return None

    text = str(model_response or "").lower()
    tagged = [
        match.group(1)
        for match in re.finditer(r"<action>(.*?)</action>", text, re.DOTALL)
    ]
    for candidate in reversed(tagged):
        projected = lookup.get(_normalize_alfworld_action(candidate))
        if projected is not None:
            return projected

    # Small models sometimes omit <action> while still placing one command after
    # </think> or on its own line. Only accept an exact admissible command; never
    # pass the surrounding reasoning text to the environment.
    segments = re.split(r"</think>|</?action>|[\r\n]+", text)
    for candidate in reversed(segments):
        projected = lookup.get(_normalize_alfworld_action(candidate))
        if projected is not None:
            return projected
    return lookup.get(_normalize_alfworld_action(text))


def extract_action_text(
    model_response: str,
    env: str = "webshop",
    available_actions: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Pull the projected env action from a model response.

    Handles BOTH formats seen in training:
    - raw response with tags: "<think>..</think><action>ACT</action>"
      (role-1 teacher query output, decoded from responses);
    - already-projected bare action (student batch model_response = env-projected
      text_actions): WebShop "click[x]"/"search[x]"; ALFWorld natural language
      "go to countertop 1" / "take apple from countertop 1".

    ``env`` selects the bare-action fallback: WebShop keys off click/search[...]
    tokens; ALFWorld actions are free-form admissible commands, so an untagged
    non-empty response IS the action (returned verbatim). Without this, ALFWorld
    bare actions returned None -> empty prefix replay -> 100% replay_mismatch.
    """
    if model_response is None:
        return None
    text = str(model_response).lower()
    if str(env) == "alfworld" and available_actions is not None:
        return _project_alfworld_action(text, available_actions)
    start, end = text.find("<action>"), text.find("</action>")
    if start >= 0 and end > start:
        return " ".join(text[start + len("<action>"): end].strip().split()) or None
    if str(env) == "searchqa":
        matches = list(re.finditer(r"<(search|answer)>.*?</\1>", text, re.DOTALL))
        if matches:
            return " ".join(matches[-1].group(0).split()) or None
        return None
    if str(env) == "alfworld":
        # Untagged bare admissible command; strip any stray <think> block first.
        ts, te = text.find("<think>"), text.find("</think>")
        if ts >= 0 and te > ts:
            text = (text[:ts] + text[te + len("</think>"):])
        return " ".join(text.split()) or None
    # WebShop bare-action fallback: match the last click[...] / search[...] token.
    m = list(re.finditer(r"(click|search)\[[^\]]*\]", text))
    if m:
        return " ".join(m[-1].group(0).split()) or None
    return None


def actions_diverge(teacher_action: Optional[str], student_action: Optional[str]) -> bool:
    """Confirmed divergence test a_τ^T ≠ a_τ^S. Both must be
    non-empty and different."""
    if not teacher_action or not student_action:
        return False
    return " ".join(teacher_action.lower().split()) != " ".join(student_action.lower().split())


def recover_prefix_actions(step_actions: Sequence[str], stall_step: int) -> List[str]:
    """The student's own actions for steps 0..stall_step-1 (the shared replay
    prefix; both branches diverge only at stall_step)."""
    return [str(a) for a in step_actions[:stall_step]]
