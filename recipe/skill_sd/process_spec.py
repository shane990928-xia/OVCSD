"""Cross-environment process specifications for offline validation.

This module deliberately has no trainer, Ray, or model dependency.  A
``ProcessSpec`` is compiled once for a task and a retrieved skill subset, then
used as a read-only verifier for process progress.  The policy never receives
the spec as prompt input.

The existing :mod:`phi_compiler` remains the task-only baseline for WebShop and
ALFWorld.  SearchQA uses evidence-oriented milestones because its environment
does not expose a physical stage machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


PROCESS_SPEC_VERSION = "v1"


@dataclass(frozen=True)
class ProcessMilestone:
    """A verifier criterion with optional provenance to retrieved skills."""

    id: str
    kind: str
    weight: float = 1.0
    param: Optional[str] = None
    terminal: bool = False
    source_skill_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessSpec:
    """Immutable, hashable task/skill-conditioned verifier specification."""

    env: str
    task_desc: str
    gamefile: str
    milestones: Tuple[ProcessMilestone, ...]
    skill_ids: Tuple[str, ...]
    spec_hash: str
    version: str = PROCESS_SPEC_VERSION

    def payload(self) -> dict[str, Any]:
        """Return the canonical payload used to compute ``spec_hash``."""
        return {
            "version": self.version,
            "env": self.env,
            "task_desc": self.task_desc,
            "gamefile": self.gamefile,
            "skill_ids": list(self.skill_ids),
            "milestones": [asdict(m) for m in self.milestones],
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.payload()
        data["spec_hash"] = self.spec_hash
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessSpec":
        milestones = tuple(
            ProcessMilestone(
                id=str(item["id"]),
                kind=str(item["kind"]),
                weight=float(item.get("weight", 1.0)),
                param=item.get("param"),
                terminal=bool(item.get("terminal", False)),
                source_skill_ids=tuple(str(x) for x in item.get("source_skill_ids", ())),
            )
            for item in data.get("milestones", ())
        )
        spec = cls(
            env=_normalize_env(str(data.get("env", ""))),
            task_desc=str(data.get("task_desc", "")),
            gamefile=str(data.get("gamefile", "")),
            milestones=milestones,
            skill_ids=tuple(sorted({str(x) for x in data.get("skill_ids", ())})),
            spec_hash=str(data.get("spec_hash", "")),
            version=str(data.get("version", PROCESS_SPEC_VERSION)),
        )
        expected = _hash_payload(spec.payload())
        if spec.spec_hash and spec.spec_hash != expected:
            raise ValueError(
                f"ProcessSpec hash mismatch: stored={spec.spec_hash}, expected={expected}"
            )
        return cls(**{**spec.__dict__, "spec_hash": expected})


def _normalize_env(env: str) -> str:
    value = str(env or "").strip().lower().replace("_", "")
    aliases = {
        "webshop": "webshop",
        "alfworld": "alfworld",
        "search": "searchqa",
        "searchqa": "searchqa",
        "searchquestionanswering": "searchqa",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported process-spec environment: {env!r}")
    return aliases[value]


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _skill_id(skill: Mapping[str, Any]) -> str:
    value = skill.get("skill_id")
    if value is None or not str(value).strip():
        raise ValueError("Every skill passed to compile_process_spec needs skill_id")
    return str(value)


def _skill_text(skill: Mapping[str, Any]) -> str:
    return " ".join(
        str(skill.get(key, ""))
        for key in ("title", "principle", "when_to_apply")
    ).lower()


# The compiler is intentionally whitelist-based.  Skill text selects a
# verifier criterion; it cannot invent a new predicate or execute code.
_INTENT_RULES: dict[str, tuple[tuple[str, tuple[str, ...], str], ...]] = {
    "webshop": (
        ("search", ("search", "query", "pagination", "refin"), "search_done"),
        ("scan", ("scan", "result page", "product link", "candidate"), "item_reached"),
        ("verify", ("verify", "validate", "constraint", "spec", "feature", "match", "confirm"), "attr_token"),
        ("option", ("variant", "option", "selector", "swatch", "size", "color", "configure"), "option_click"),
        ("price", ("price", "budget", "cost"), "price_ok"),
        ("buy", ("buy", "purchase", "checkout", "commit"), "purchased"),
        ("backtrack", ("loop", "abort", "fallback", "exit", "return"), "search_done"),
    ),
    "alfworld": (
        ("explore", ("explor", "search", "surface", "container", "room", "location", "hunt"), "explored"),
        ("acquire", ("acquire", "pick up", "grab", "fetch", "take"), "picked"),
        ("navigate", ("destination", "navigate", "transport", "delivery", "go to", "spatial"), "arrived"),
        ("state", ("heat", "cool", "clean", "lamp", "light", "state", "transform", "appliance"), "state_changed"),
        ("place", ("place", "put", "drop", "holding"), "placed"),
        ("count", ("count", "multiple", "two", "instance"), "counted"),
    ),
    "searchqa": (
        ("decompose", ("decompos", "multi-hop", "multi part", "sub-question", "attribute-chain"), "subquestion_covered"),
        ("query", ("query", "search", "retrieval", "precision", "freshness", "latest"), "search_issued"),
        ("evidence", ("source-backed", "evidence", "assertion", "answer supported"), "evidence_obtained"),
        ("crosscheck", ("cross-check", "multiple source", "disagree"), "source_diversity"),
        ("ambiguity", ("ambigu", "entity resolution"), "ambiguity_resolved"),
        ("compare", ("compar", "quantitative", "versus", "difference"), "comparison_formed"),
    ),
}


def skill_intents(env: str, skill: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return one or two stage-specific intents for one skill.

    The title and ``when_to_apply`` field carry more reliable stage information
    than incidental words in the principle.  A weighted classifier prevents a
    generic word such as ``answer`` or ``object`` from activating unrelated
    verifiers for every skill in a bank.
    """
    env = _normalize_env(env)
    title = str(skill.get("title", "")).lower()
    when = str(skill.get("when_to_apply", "")).lower()
    principle = str(skill.get("principle", "")).lower()
    scored: list[tuple[float, int, str, str]] = []
    for order, (intent, keywords, kind) in enumerate(_INTENT_RULES[env]):
        score = 0.0
        for keyword in keywords:
            if keyword in title:
                score += 4.0
            if keyword in when:
                score += 2.0
            if keyword in principle:
                score += 1.0
        if score > 0.0:
            scored.append((score, order, intent, kind))
    if scored:
        scored.sort(key=lambda item: (-item[0], item[1]))
        best = scored[0][0]
        # Keep a second intent only when the title explicitly expresses a
        # compound stage (for example, "Check Variant Price").
        selected = [item for item in scored if item[0] >= best * 0.75][:2]
        return tuple((item[2], item[3]) for item in selected)
    fallback = {
        "webshop": ("generic_progress", "item_reached"),
        "alfworld": ("generic_progress", "explored"),
        "searchqa": ("generic_progress", "evidence_obtained"),
    }[env]
    return (fallback,)


def _legacy_milestones(env: str, task_desc: str, gamefile: str) -> list[ProcessMilestone]:
    """Convert the existing task-only compiler output to this API."""
    if env == "searchqa":
        return [
            ProcessMilestone("search_issued", "search_issued", 1.0),
            ProcessMilestone("evidence_obtained", "evidence_obtained", 1.0),
            ProcessMilestone("answer_submitted", "answer_submitted", 1.0),
            ProcessMilestone("answer_correct", "answer_correct", 2.0, terminal=True),
        ]

    # Lazy import avoids a module cycle and keeps this module dependency-free
    # for callers that only use SearchQA.
    from recipe.skill_sd.phi_compiler import compile_milestones

    legacy = compile_milestones(env, task_desc=task_desc, gamefile=gamefile)
    return [
        ProcessMilestone(
            id=m.id,
            kind=m.kind,
            weight=float(m.weight),
            param=m.param,
            terminal=bool(m.terminal),
        )
        for m in legacy
    ]


def compile_process_spec(
    env: str,
    *,
    task_desc: str = "",
    gamefile: str = "",
    skills: Optional[Iterable[Mapping[str, Any]]] = None,
) -> ProcessSpec:
    """Compile a stable task/skill-conditioned verifier specification.

    ``skills=None`` and ``skills=[]`` are equivalent task-only baselines.  The
    returned spec is independent of input ordering, which is important when
    retrieval uses a tie-breaker or dropout.
    """
    normalized_env = _normalize_env(env)
    task_desc = str(task_desc or "")
    gamefile = str(gamefile or "")
    normalized_skills = sorted(list(skills or ()), key=_skill_id)

    milestones = _legacy_milestones(normalized_env, task_desc, gamefile)
    skill_ids = set()
    sources_by_index: dict[int, set[str]] = {}
    extra_sources: dict[tuple[str, str, Optional[str], bool], set[str]] = {}
    votes_by_index: dict[int, float] = {}
    extra_votes: dict[tuple[str, str, Optional[str], bool], float] = {}
    for skill in normalized_skills:
        sid = _skill_id(skill)
        skill_ids.add(sid)
        assignments: set[tuple[str, Any]] = set()
        for intent, kind in skill_intents(normalized_env, skill):
            for target in _ground_skill_intent(
                normalized_env,
                intent,
                kind,
                task_desc=task_desc,
                milestones=milestones,
                skill=skill,
            ):
                target_kind, target_param, extra_id = target
                if not _skill_target_is_discriminative(normalized_env, target_kind):
                    # Keep ubiquitous stage arrivals in the task-only graph;
                    # letting them consume skill mass changes only the
                    # denominator and rewards short trajectories by accident.
                    continue
                matched = False
                for index, milestone in enumerate(milestones):
                    if milestone.kind != target_kind:
                        continue
                    if target_param is not None and milestone.param != target_param:
                        continue
                    assignments.add(("base", index))
                    matched = True
                if not matched and extra_id:
                    key = (extra_id, target_kind, target_param, False)
                    assignments.add(("extra", key))

        # Every applicable skill contributes one vote in total, split across
        # its grounded criteria. This keeps the total skill mass bounded while
        # allowing different skill subsets to redistribute Phi's emphasis.
        if assignments:
            vote = 1.0 / len(assignments)
            for target_type, target in assignments:
                if target_type == "base":
                    index = int(target)
                    sources_by_index.setdefault(index, set()).add(sid)
                    votes_by_index[index] = votes_by_index.get(index, 0.0) + vote
                else:
                    key = target
                    extra_sources.setdefault(key, set()).add(sid)
                    extra_votes[key] = extra_votes.get(key, 0.0) + vote

    # A criterion is represented once even when several skills recommend it.
    # The source IDs retain the S -> spec path without multiplying an identical
    # Boolean predicate by the number of retrieved skills.
    total_votes = sum(votes_by_index.values()) + sum(extra_votes.values())
    vote_scale = 2.0 / total_votes if total_votes > 0.0 else 0.0
    for index, sources in sources_by_index.items():
        milestone = milestones[index]
        milestones[index] = replace(
            milestone,
            weight=float(milestone.weight) + votes_by_index[index] * vote_scale,
            source_skill_ids=tuple(sorted(sources)),
        )
    for (mid, kind, param, terminal), sources in extra_sources.items():
        milestones.append(
            ProcessMilestone(
                id=mid,
                kind=kind,
                weight=extra_votes[(mid, kind, param, terminal)] * vote_scale,
                param=param,
                terminal=terminal,
                source_skill_ids=tuple(sorted(sources)),
            )
        )

    # Canonical ordering makes the hash invariant to retrieval ordering.
    milestones = sorted(
        milestones,
        key=lambda m: (m.id, m.kind, m.param or "", m.source_skill_ids),
    )
    skill_tuple = tuple(sorted(skill_ids))
    provisional = ProcessSpec(
        env=normalized_env,
        task_desc=task_desc,
        gamefile=gamefile,
        milestones=tuple(milestones),
        skill_ids=skill_tuple,
        spec_hash="",
    )
    return ProcessSpec(**{**provisional.__dict__, "spec_hash": _hash_payload(provisional.payload())})


def _skill_target_is_discriminative(env: str, kind: str) -> bool:
    non_discriminative = {
        "webshop": {"search_done", "item_reached"},
        "searchqa": {"search_issued"},
    }
    return kind not in non_discriminative.get(env, set())


def _ground_skill_intent(
    env: str,
    intent: str,
    kind: str,
    *,
    task_desc: str,
    milestones: Sequence[ProcessMilestone],
    skill: Mapping[str, Any],
) -> tuple[tuple[str, Optional[str], Optional[str]], ...]:
    """Map a skill intent to an applicable verifier from the whitelist.

    The returned tuple is ``(kind, param, extra_id)``.  ``extra_id`` is
    only used when the task-only spec does not already contain that criterion.
    Applicability checks are especially important for SearchQA: unconditional
    cross-check and ambiguity milestones reward extra searches on questions
    that do not need them.
    """
    if env == "webshop":
        return ((kind, None, None),)

    if env == "alfworld":
        if kind == "state_changed":
            state_kinds = ("heated", "cooled", "cleaned", "turned_on")
            task_state = next(
                (candidate for candidate in state_kinds if any(m.kind == candidate for m in milestones)),
                None,
            )
            if task_state is None:
                return ()
            text = _skill_text(skill)
            named_states = {
                "heated": ("heat", "microwave"),
                "cooled": ("cool", "fridge"),
                "cleaned": ("clean", "sink"),
                "turned_on": ("lamp", "light", "toggle"),
            }
            explicit = {
                candidate
                for candidate, keywords in named_states.items()
                if any(keyword in text for keyword in keywords)
            }
            if explicit and task_state not in explicit:
                return ()
            return ((task_state, None, None),)
        if kind in {"picked", "placed"}:
            return ((kind, None, None),)
        # Exploration/navigation advice is useful for prompting but has no
        # target-specific completion event in the current task-only graph.
        return ()

    question = task_desc.lower()
    if kind == "evidence_obtained":
        return (("answer_supported", None, "skill::answer_supported"),)
    if kind == "search_issued":
        return (("search_issued", None, None),)
    if kind == "source_diversity":
        temporal = ("current", "latest", "next", "recent", "today", "now")
        if any(token in question for token in temporal):
            return ((kind, None, "skill::source_diversity"),)
        return ()
    if kind == "ambiguity_resolved":
        ambiguity = ("which one", "same name", "refer to", " or ")
        if any(token in question for token in ambiguity):
            return ((kind, None, "skill::ambiguity_resolved"),)
        return ()
    if kind == "comparison_formed":
        comparison = ("compare", "difference", " versus ", " vs ", "earlier", "later", "older", "younger", "more than", "less than")
        if any(token in question for token in comparison):
            return ((kind, None, "skill::comparison_formed"),)
        return ()
    if kind == "subquestion_covered":
        complex_question = (" and ", "both ", "multi-hop", "relationship between")
        if any(token in question for token in complex_question):
            return ((kind, None, "skill::subquestion_covered"),)
        return ()
    return ()


def _contains_any(text: str, values: Sequence[str]) -> bool:
    return any(value in text for value in values)


def _milestone_done(
    spec: ProcessSpec,
    milestone: ProcessMilestone,
    observations: Sequence[str],
    actions: Sequence[str],
    terminal_reward: Optional[float],
) -> bool:
    if milestone.terminal and terminal_reward is not None:
        threshold = 1.0 if spec.env == "searchqa" else 10.0
        return float(terminal_reward) >= threshold

    obs = " [OBSSEP] ".join(str(x or "") for x in observations).lower()
    acts = [str(x or "").lower() for x in actions]
    act_text = " ".join(acts)
    kind = milestone.kind

    if spec.env == "webshop":
        if kind == "search_done":
            return "total results:" in obs
        if kind == "item_reached":
            return "buy now" in obs or "click[buy now]" in act_text
        if kind == "option_click":
            param = str(milestone.param or "").lower()
            tokens = [
                token
                for token in re.split(r"[|,/ ]+", param)
                if len(token) >= 2
            ]
            if not tokens:
                return False
            return any(
                action.startswith("click[") and any(token in action for token in tokens)
                for action in acts
            )
        if kind == "price_ok":
            try:
                cap = float(milestone.param) if milestone.param is not None else 1e9
            except (TypeError, ValueError):
                cap = 1e9
            prices = [float(x) for x in re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", obs)]
            return any(price <= cap for price in prices)
        if kind == "attr_token":
            return bool(milestone.param and re.search(r"\b" + re.escape(milestone.param.lower()) + r"\b", obs))
        if kind == "purchased":
            return "click[buy now]" in act_text
        return False

    if spec.env == "alfworld":
        if kind == "explored":
            return _contains_any(obs, ("arrive", "room", "shelf", "cabinet", "location"))
        if kind == "arrived":
            target = re.sub(r"[^a-z]", "", str(milestone.param or "").lower())
            if not target:
                return False
            return bool(
                re.search(r"(?:arrive at|go to|near)\s+(?:the\s+)?" + re.escape(target), obs + " " + act_text)
            )
        if kind == "picked":
            target = re.sub(r"[^a-z]", "", str(milestone.param or "").lower())
            if not target:
                return "you pick up the" in obs
            return bool(
                re.search(r"you pick up the\s+" + re.escape(target), obs)
                or re.search(r"(?:take|pick up)\s+(?:the\s+)?" + re.escape(target), act_text)
            )
        if kind in {"heated", "cooled", "cleaned", "state_changed"}:
            event = {
                "heated": "you heat the",
                "cooled": "you cool the",
                "cleaned": "you clean the",
            }.get(kind)
            if event:
                return event in obs
            return _contains_any(obs, ("you heat the", "you cool the", "you clean the"))
        if kind == "turned_on":
            return "you turn on the" in obs
        if kind == "placed":
            target = re.sub(r"[^a-z]", "", str(milestone.param or "").lower())
            if not target:
                return "you put the" in obs or "you move the" in obs
            pattern = r"you (?:put|move) the [a-z0-9 ]*?(?:in|on|to)[a-z0-9 ]*?" + re.escape(target)
            return bool(re.search(pattern, obs))
        if kind == "counted":
            return "two" in obs or "count" in obs
        return False

    # SearchQA is evidence-oriented rather than stage-oriented.
    if kind == "search_issued":
        return bool(re.search(r"<search>.*?</search>", act_text, re.DOTALL))
    if kind == "evidence_obtained":
        return "<information>" in obs or "information" in obs
    if kind == "source_diversity":
        return obs.count("<information>") >= 2
    if kind == "subquestion_covered":
        question_terms = [
            token for token in re.findall(r"[a-z]{4,}", spec.task_desc.lower())
            if token not in {"what", "when", "where", "which", "does", "with", "from"}
        ]
        return any(re.search(r"\b" + re.escape(token) + r"\b", obs) for token in question_terms)
    if kind == "ambiguity_resolved":
        return _contains_any(obs, ("clarif", "specifically", "refers to"))
    if kind == "comparison_formed":
        return _contains_any(obs, ("compared", "versus", "difference", "earlier", "later"))
    if kind == "answer_submitted":
        return bool(re.search(r"<answer>.*?</answer>", act_text, re.DOTALL))
    if kind == "answer_supported":
        answers = re.findall(r"<answer>(.*?)</answer>", act_text, re.DOTALL)
        if not answers or "<information>" not in obs:
            return False
        answer = re.sub(r"[^a-z0-9]", "", answers[-1].lower())
        evidence = re.sub(r"[^a-z0-9]", "", obs.lower())
        return bool(answer) and answer in evidence
    if kind == "answer_correct":
        return False
    return False


def process_phi_value(
    spec: ProcessSpec,
    observations: Sequence[str],
    actions: Sequence[str],
    terminal_reward: Optional[float] = None,
) -> float:
    """Evaluate normalized process progress for a compiled spec."""
    total_weight = sum(float(m.weight) for m in spec.milestones) or 1.0
    return sum(
        float(m.weight)
        for m in spec.milestones
        if _milestone_done(spec, m, observations, actions, terminal_reward)
    ) / total_weight


def process_phi_curve(
    spec: ProcessSpec,
    observations: Sequence[str],
    actions: Sequence[str],
    terminal_reward: Optional[float] = None,
) -> list[float]:
    """Return cumulative verifier progress at every trajectory step."""
    curve = []
    for index in range(min(len(observations), len(actions))):
        reward = terminal_reward if index == min(len(observations), len(actions)) - 1 else None
        curve.append(process_phi_value(spec, observations[: index + 1], actions[: index + 1], reward))
    return curve


def process_delta_phi(curve: Sequence[float], gamma: float = 1.0) -> list[float]:
    """Compute the potential-based transition increments."""
    return [float(gamma) * curve[i + 1] - curve[i] for i in range(max(len(curve) - 1, 0))]
