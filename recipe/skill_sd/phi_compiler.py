"""Compute observable task milestones for environment trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence


# Environment success thresholds (episode reward at/above => task success).
WEBSHOP_SUCCESS_REWARD = 10.0
ALFWORLD_SUCCESS_REWARD = 10.0


# Generic goal-token extraction (env-agnostic; see _extract_goal_tokens). Filters
# instruction boilerplate + function words so only salient goal-attribute tokens
# remain. This is deliberately NOT a domain schema (no WebShop-specific attribute
# categories) -- the instruction's own content words are the specification, and
# "is this word present in the observation" is a generic verifiable check that
# works for any text instruction+observation environment.
_GOAL_STOPWORDS = frozenset({
    "i", "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "that", "is", "are", "am", "it", "its", "at", "as", "be", "by", "me", "my",
    "want", "wanna", "find", "buy", "get", "got", "have", "has", "some", "any",
    "what", "which", "who", "where", "how", "this", "these", "those", "there",
    "price", "priced", "lower", "less", "than", "under", "below", "dollar",
    "dollars", "cost", "costs", "need", "needs", "please", "would", "like",
    "can", "could", "should", "will", "looking", "look", "one", "ones", "item",
    "product", "products", "purchase", "show", "give", "from", "but", "not",
})


@dataclass(frozen=True)
class Milestone:
    """One compiled milestone with its verifiable criterion.

    ``terminal=True`` marks the goal milestone that should be judged by the
    environment reward (not observation regex) when a terminal reward is known.
    """

    id: str
    weight: float
    kind: str
    param: Optional[str] = None
    terminal: bool = False


# --------------------------------------------------------------------------- #
# WebShop compiler (stage state-machine; Tier-1/2 verifiable criteria)
# --------------------------------------------------------------------------- #
def _extract_goal_tokens(task_desc: str, min_len: int = 3, max_tokens: int = 12) -> List[str]:
    """Extract salient content tokens from a goal instruction (env-agnostic).

    Strips the price clause and currency, splits on non-alphanumerics, drops
    function words (``_GOAL_STOPWORDS``) and short tokens, dedupes preserving
    order. These tokens become fine-grained "attribute present in observation"
    milestones so Φ can distinguish a good item match from a poor one -- the
    signal the coarse 4-milestone Φ was blind to. Generic by construction: no
    domain schema, just the instruction's own words vs the observation text.
    """
    if not task_desc:
        return []
    text = re.sub(r"price\s+lower\s+than.*", " ", str(task_desc).lower())
    text = re.sub(r"\$?\s*[0-9]+(?:\.[0-9]+)?\s*(?:dollars?)?", " ", text)
    seen, out = set(), []
    for tok in re.split(r"[^a-z0-9]+", text):
        if len(tok) >= min_len and tok not in _GOAL_STOPWORDS and tok not in seen:
            seen.add(tok)
            out.append(tok)
        if len(out) >= max_tokens:
            break
    return out


def _webshop_compile(task_desc: str) -> List[Milestone]:
    price = None
    mp = re.search(r"price lower than\s*\$?\s*([0-9.]+)", task_desc, re.I)
    if mp:
        try:
            price = float(mp.group(1))
        except ValueError:
            price = None
    color = None
    mc = re.search(r"color:\s*([^,]+?)(?:,| and | size:|\.|$)", task_desc, re.I)
    if mc:
        color = mc.group(1).strip().lower()
    size = None
    ms = re.search(r"size:\s*([^,]+?)(?:,| and | price|\.|$)", task_desc, re.I)
    if ms:
        size = ms.group(1).strip().lower()

    milestones = [
        Milestone("search_done", 1.0, "search_done"),
        Milestone("item_reached", 1.0, "item_reached"),
    ]
    if color:
        milestones.append(Milestone("color_selected", 1.0, "option_click", color))
    if size:
        milestones.append(Milestone("size_selected", 1.0, "option_click", size))
    if price is not None:
        milestones.append(Milestone("price_ok", 1.0, "price_ok", str(price)))
    # Fine-grained attribute coverage: each salient goal token is a small-weight
    # milestone that fires when the token appears in the observation (i.e. the
    # reached item page mentions it). This makes Φ rise more for an item matching
    # many goal attributes than one matching few -- the item-quality gradient the
    # coarse milestones missed. Skips tokens already covered by color/size params
    # to avoid double-counting. Weight 0.5 keeps the stage milestones dominant.
    covered = set()
    for c in (color, size):
        if c:
            covered.update(t for t in re.split(r"[|,/ ]+", c) if len(t) >= 3)
    for tok in _extract_goal_tokens(task_desc):
        if tok in covered:
            continue
        milestones.append(Milestone(f"attr::{tok}", 0.5, "attr_token", tok))
    # Terminal milestone: judged by env reward when available (see module docstring).
    milestones.append(Milestone("purchased", 2.0, "purchased", terminal=True))
    return milestones


def _webshop_done(ms: Milestone, obs_hist: Sequence[str], act_hist: Sequence[str]) -> bool:
    kind = ms.kind
    obs_all = " [OBSSEP] ".join(obs_hist).lower()
    if kind == "search_done":
        return "total results:" in obs_all
    if kind == "item_reached":
        return any(a.startswith("click[") and "buy now" in a for a in act_hist) or "buy now" in obs_all
    if kind == "option_click":
        param = str(ms.param or "")
        toks = [t for t in re.split(r"[|,/ ]+", param) if len(t) >= 2]
        return any(any(t in a for t in toks) and a.startswith("click[") for a in act_hist)
    if kind == "price_ok":
        try:
            cap = float(ms.param) if ms.param is not None else 1e9
        except ValueError:
            cap = 1e9
        for o in obs_hist:
            for pm in re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", o):
                try:
                    if float(pm) <= cap:
                        return True
                except ValueError:
                    pass
        return False
    if kind == "attr_token":
        # Generic: goal token present in the observation text (the item page /
        # results mention this attribute). Word-boundary match to avoid spurious
        # substring hits (e.g. "fit" inside "outfit").
        tok = str(ms.param or "")
        return bool(tok) and re.search(r"\b" + re.escape(tok) + r"\b", obs_all) is not None
    if kind == "purchased":
        return any("buy now" in a and a.startswith("click[") for a in act_hist)
    return False


# --------------------------------------------------------------------------- #
# ALFWorld compiler (Tier-1 state-change phrases; terminal = reward)
# --------------------------------------------------------------------------- #
_ALFWORLD_TYPE_RE = re.compile(
    r"(pick_and_place(?:_simple)?|pick_two_obj_and_place|look_at_obj_in_light"
    r"|pick_heat_then_place_in_recep|pick_cool_then_place_in_recep"
    r"|pick_clean_then_place_in_recep)-([A-Za-z]+)-[^-/]*-([A-Za-z]+)"
)


def _alfworld_compile(gamefile: str) -> List[Milestone]:
    g = gamefile.lower()
    m = _ALFWORLD_TYPE_RE.search(g)
    ttype = m.group(1) if m else "pick_and_place"
    obj = m.group(2).lower() if m else None
    recep = m.group(3).lower() if m else None

    milestones = [Milestone("picked", 1.0, "picked", obj)]
    if "heat" in ttype:
        milestones.append(Milestone("heated", 1.5, "heated", obj))
    elif "cool" in ttype:
        milestones.append(Milestone("cooled", 1.5, "cooled", obj))
    elif "clean" in ttype:
        milestones.append(Milestone("cleaned", 1.5, "cleaned", obj))
    elif "look_at_obj_in_light" in ttype:
        milestones.append(Milestone("lit", 1.5, "turned_on"))
    # Terminal milestone: judged by env reward.
    milestones.append(Milestone("placed", 2.0, "placed", recep, terminal=True))
    return milestones


def _alfworld_done(ms: Milestone, obs_hist: Sequence[str], act_hist: Sequence[str]) -> bool:
    kind = ms.kind
    obs_all = " [OBSSEP] ".join(obs_hist).lower()
    obj_tok = re.sub(r"[^a-z]", "", str(ms.param or ""))
    if kind == "picked":
        return bool(re.search(r"you pick up the %s" % re.escape(obj_tok), obs_all)) if obj_tok else ("you pick up the" in obs_all)
    if kind == "heated":
        return "you heat the" in obs_all
    if kind == "cooled":
        return "you cool the" in obs_all
    if kind == "cleaned":
        return "you clean the" in obs_all
    if kind == "turned_on":
        return "you turn on the" in obs_all
    if kind == "placed":
        recep_tok = re.sub(r"[^a-z]", "", str(ms.param or ""))
        if not recep_tok:
            return ("you put the" in obs_all) or ("you move the" in obs_all)
        pat = r"you (?:put|move) the [a-z0-9 ]*?(?:in|on|to)[a-z0-9 ]*?%s" % re.escape(recep_tok)
        return bool(re.search(pat, obs_all))
    return False


# --------------------------------------------------------------------------- #
# Unified interface
# --------------------------------------------------------------------------- #
def compile_milestones(env: str, task_desc: str = "", gamefile: str = "") -> List[Milestone]:
    """Compile the milestone graph G_{x,S} for a task.

    WebShop uses ``task_desc`` (the "Find me ... price lower than N" instruction);
    ALFWorld uses ``gamefile`` (task type + target object/receptacle).
    """
    if env == "webshop":
        return _webshop_compile(task_desc)
    if env == "alfworld":
        return _alfworld_compile(gamefile or task_desc)
    raise ValueError(f"Unknown env for phi compiler: {env}")


def _success_reward(env: str) -> float:
    return WEBSHOP_SUCCESS_REWARD if env == "webshop" else ALFWORLD_SUCCESS_REWARD


def _milestone_done(
    env: str,
    ms: Milestone,
    obs_hist: Sequence[str],
    act_hist: Sequence[str],
    terminal_reward: Optional[float],
) -> bool:
    # Terminal milestone judged by env reward when known.
    if ms.terminal and terminal_reward is not None:
        return float(terminal_reward) >= _success_reward(env)
    done_fn = _webshop_done if env == "webshop" else _alfworld_done
    return done_fn(ms, obs_hist, act_hist)


def phi_value(
    env: str,
    milestones: Sequence[Milestone],
    obs_hist: Sequence[str],
    act_hist: Sequence[str],
    terminal_reward: Optional[float] = None,
) -> float:
    """Cumulative Φ(h) = Σ_j w_j · done_j(h) / Σ_j w_j, in [0, 1].

    ``terminal_reward`` (episode reward if the trajectory has ended, else None)
    is used to judge the terminal milestone; intermediate milestones always use
    observation grounding.
    """
    total_w = sum(m.weight for m in milestones) or 1.0
    val = sum(m.weight for m in milestones if _milestone_done(env, m, obs_hist, act_hist, terminal_reward))
    return val / total_w


def phi_curve(
    env: str,
    milestones: Sequence[Milestone],
    obs: Sequence[str],
    acts: Sequence[str],
    terminal_reward: Optional[float] = None,
) -> List[float]:
    """Per-step Φ over a trajectory (monotone cumulative).

    ``terminal_reward`` only affects the final step (the terminal milestone can
    only be satisfied once the episode is done).
    """
    curve = []
    n = len(obs)
    for t in range(n):
        oh, ah = obs[: t + 1], acts[: t + 1]
        # Terminal reward applies only at the last step of the trajectory.
        tr = terminal_reward if t == n - 1 else None
        curve.append(phi_value(env, milestones, oh, ah, tr))
    return curve


def delta_phi(curve: Sequence[float], gamma: float = 1.0) -> List[float]:
    """ΔΦ_t = γ·Φ(h_{t+1}) − Φ(h_t) for adjacent states."""
    return [gamma * curve[t + 1] - curve[t] for t in range(len(curve) - 1)]


def shape_rewards_with_phi(
    *,
    env: str,
    traj_uids: Sequence[str],
    step_idxs: Sequence[int],
    observations: Sequence[str],
    actions: Sequence[str],
    rewards: Sequence[float],
    task_of_traj: dict,
    terminal_reward_of_traj: Optional[dict] = None,
    eta: float = 1.0,
    gamma: float = 0.95,
):
    """Add the potential-based process reward r^proc = γΦ(h_{t+1}) − Φ(h_t) to the
    per-step environment rewards.

    Rows are per-(traj, step). We group by traj_uid, order by step_idx, compute
    the Φ curve for that trajectory (terminal milestone uses the trajectory's
    terminal reward), derive ΔΦ per step, and add ``eta * ΔΦ_t`` to step t's
    reward. Returns (new_rewards: list[float], metrics: dict).

    ``task_of_traj[traj_uid]`` = the task descriptor (WebShop instruction /
    ALFWorld gamefile) used to compile milestones.
    ``terminal_reward_of_traj[traj_uid]`` = episode reward (for the terminal
    milestone); None/absent means the trajectory is treated as non-terminal.

    Potential-based shaping is policy-invariant: the extra reward telescopes over
    a trajectory, so it does not change the optimal policy — only densifies credit.
    """
    n = len(traj_uids)
    new_rewards = [float(r) for r in rewards]

    # Group row indices by trajectory, ordered by step.
    by_traj: dict = {}
    for i in range(n):
        by_traj.setdefault(str(traj_uids[i]), []).append(i)
    for rows in by_traj.values():
        rows.sort(key=lambda i: int(step_idxs[i]))

    terminal_reward_of_traj = terminal_reward_of_traj or {}
    all_phi: List[float] = []
    final_phi: List[float] = []
    all_rproc: List[float] = []
    for tuid, rows in by_traj.items():
        task = task_of_traj.get(tuid, "")
        if env == "alfworld":
            milestones = compile_milestones(env, gamefile=str(task))
        else:
            milestones = compile_milestones(env, task_desc=str(task))
        obs_seq = [str(observations[i]) if observations[i] is not None else "" for i in rows]
        act_seq = [str(actions[i]) if actions[i] is not None else "" for i in rows]
        term_r = terminal_reward_of_traj.get(tuid)
        curve = phi_curve(env, milestones, obs_seq, act_seq, terminal_reward=term_r)
        all_phi.extend(curve)
        if curve:
            final_phi.append(curve[-1])
        dphi = delta_phi(curve, gamma=gamma)  # len = len(rows)-1, index t -> step t
        for t, d in enumerate(dphi):
            row = rows[t]
            new_rewards[row] += eta * d
            all_rproc.append(d)

    metrics = {
        "cpsd/phi_mean": float(sum(all_phi) / len(all_phi)) if all_phi else 0.0,
        "cpsd/phi_final_mean": float(sum(final_phi) / len(final_phi)) if final_phi else 0.0,
        "cpsd/rproc_mean": float(sum(all_rproc) / len(all_rproc)) if all_rproc else 0.0,
        "cpsd/rproc_nonzero_frac": float(sum(1 for d in all_rproc if abs(d) > 1e-9) / len(all_rproc)) if all_rproc else 0.0,
    }
    return new_rewards, metrics
