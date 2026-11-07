"""Student rollouts, replay-verified intervention and first-divergence grouping."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean, pstdev


def canonical(text):
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class Action:
    command: str
    response_ids: tuple
    positions: tuple  # Generated token indices belonging to the executed command.


@dataclass(frozen=True)
class Step:
    observation: str
    action: Action
    following: str


@dataclass
class Episode:
    task: str
    goal: str
    initial: str
    steps: list
    reward: float
    score: float


@dataclass
class Site:
    depth: int  # Absolute depth, including the replayed prefix.
    teacher: Episode
    members: list
    skills: str

    @property
    def suffix(self):
        return self.teacher.steps[self.depth + 1:]


def rollout(env, policy, task, horizon, counts, *, prefix=None, depth=0, skills=None):
    """Reset to the task, verify every transition, then execute the policy."""
    initial, goal = env.reset(task)
    steps = []
    observation = initial
    if prefix is not None:
        if (canonical(initial), canonical(goal)) != (canonical(prefix.initial), canonical(prefix.goal)):
            return None
        for previous in prefix.steps[:depth]:
            observation, _, done = env.step(previous.action.command)
            counts["replay_actions"] += 1
            if done or canonical(observation) != canonical(previous.following):
                return None
            steps.append(previous)

    source = "student_actions" if skills is None else "teacher_actions"
    for _ in range(depth, horizon):
        action = policy.act(goal, steps, observation, skills)
        following, score, done = env.step(action.command)
        counts[source] += 1
        steps.append(Step(observation, action, following))
        observation = following
        if done:
            return Episode(task, goal, initial, steps, 10.0 * (score == 1.0), score)
    return Episode(task, goal, initial, steps, 0.0, 0.0)


def advantages(group):
    rewards = [episode.reward for episode in group]
    mean, std = fmean(rewards), pstdev(rewards)
    return [(r - mean) / (std + 1e-6) for r in rewards], max(rewards) < 10 and std <= 1e-8


def prefix_nodes(group, depth_limit):
    """A prefix is a path of canonical (observation, action, observation) triples."""
    members = defaultdict(list)
    for index, episode in enumerate(group):
        prefix = ()
        for depth, step in enumerate(episode.steps):
            if depth > depth_limit:
                break
            members[prefix].append(index)
            prefix += ((canonical(step.observation), canonical(step.action.command),
                        canonical(step.following)),)
        # A terminal member of a shared prefix has no valid next action (Eq. 10).
        if depth_limit >= len(episode.steps):
            members[prefix].append(index)
    eligible = {}
    for path, ids in members.items():
        depth = len(path)
        if not path or (len(ids) >= 2 and all(
            depth < len(group[i].steps) and group[i].steps[depth].action.positions for i in ids
        )):
            eligible[path] = ids
    return eligible


def first_difference(teacher, student, start):
    for d in range(start, min(len(teacher.steps), len(student.steps))):
        left, right = teacher.steps[d], student.steps[d]
        if canonical(left.observation) != canonical(right.observation):
            break
        if canonical(left.action.command) != canonical(right.action.command):
            if left.action.positions and right.action.positions:
                return d
            break
        if canonical(left.following) != canonical(right.following):
            break
    return None


def intervene(env, policy, group, horizon, depth_limit, skills, counts):
    """Retry each verified prefix up to its node budget before backing off."""
    if not advantages(group)[1]:
        return []
    if any(e.task != group[0].task or canonical(e.initial) != canonical(group[0].initial)
           or canonical(e.goal) != canonical(group[0].goal) for e in group):
        raise ValueError("The rollout group contains different initial states or task identities")
    nodes = prefix_nodes(group, depth_limit)
    ancestors = {path[:d] for path in nodes for d in range(len(path))}
    pending = set(nodes) - ancestors
    tried, covered = set(), set()
    sites = []
    while pending:
        path = min(pending, key=lambda p: (-len(p), p))
        pending.remove(path)
        tried.add(path)
        members = [i for i in nodes[path] if i not in covered]
        if len(members) < (2 if path else 1):
            continue
        episode = group[members[0]]
        verified = False
        for _ in range(4 if path else 1):
            branch = rollout(env, policy, episode.task, horizon, counts, prefix=episode,
                             depth=len(path), skills=skills)
            if branch is None:
                counts["replay_mismatches"] += 1
                break
            counts["teacher_attempts"] += 1
            if branch.reward == 10:
                matched = defaultdict(list)
                for i in members:
                    d = first_difference(branch, group[i], len(path))
                    if d is not None:
                        matched[d].append(group[i])
                        covered.add(i)
                sites.extend(Site(d, branch, rows, skills) for d, rows in matched.items())
                verified = True
                break
        if verified:
            continue
        for d in range(len(path) - 1, -1, -1):
            parent = path[:d]
            if parent in nodes and parent not in tried:
                pending.add(parent)
                break
    return sites


def collect(env, policy, tasks, horizon, depth_limit, retrieve, k=8):
    counts = Counter()
    # Finish the entire skill-free batch before any privileged generation.
    groups = [[rollout(env, policy, task, horizon, counts) for _ in range(k)] for task in tasks]
    sites = []
    for group in groups:
        if advantages(group)[1]:
            sites.extend(intervene(env, policy, group, horizon, depth_limit,
                                   retrieve(group[0].goal), counts))
    return groups, sites, counts
