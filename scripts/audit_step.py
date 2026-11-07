"""Audit a completed native ALFWorld rollout using its per-trajectory and replay records."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json


def audit(run_dir, step):
    trajectories = defaultdict(dict)
    jobs, attempts, pairs, signals, suffix = [], [], [], [], []
    for file in sorted(run_dir.rglob("records_*.jsonl.gz")):
        with gzip.open(file, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                if record["global_step"] != step:
                    continue
                data = record["payload"]
                kind = record["record_type"]
                if kind == "train_student_trajectory":
                    task = data["task_id"]
                    key = data["traj_uid"]
                    if key in trajectories[task]:
                        raise ValueError(f"Duplicate trajectory {key}; use a fresh run directory")
                    trajectories[task][key] = float(data["episode_reward"])
                elif kind == "teacher_rescue_job":
                    jobs.append(data)
                elif kind == "teacher_rescue_attempt":
                    attempts.append(data)
                elif kind == "rescue_training_signal":
                    signals.append(data["local_group_uid"])
                elif kind == "rescue_pair":
                    pairs.append(data)
                elif kind == "rescue_suffix_kl_example":
                    suffix.append((data["group_uid"], data["environment_step"]))
    if len(trajectories) != 8 or any(len(group) != 8 for group in trajectories.values()):
        raise ValueError("Expected eight complete task groups of eight")
    if not all(r in (0, 10) for g in trajectories.values() for r in g.values()):
        raise ValueError("Unexpected ALFWorld reward")
    all_failed = {task for task, group in trajectories.items() if max(group.values()) == 0}
    used = Counter()
    for job in jobs:
        if job["task_id"] not in all_failed:
            raise ValueError("Teacher job on a task with a successful student")
        used[job["task_id"], job["node_uid"], bool(job["is_root"])] += 1
    if any(n > (1 if root else 4) for (_, _, root), n in used.items()):
        raise ValueError("Teacher attempt budget exceeded")
    def attempt_id(record):
        return record["task_id"], record["group_uid"], record["node_uid"], record["attempt"]

    scheduled = Counter(map(attempt_id, jobs))
    if scheduled != Counter(map(attempt_id, attempts)) or any(n != 1 for n in scheduled.values()):
        raise ValueError("Teacher jobs and recorded outcomes do not match one to one")
    accepted = set()
    for a in attempts:
        if a["success"]:
            if not (a["replay_verified"] and a["done"] and a["terminal_reward"] == 10):
                raise ValueError("Unverified successful rescue")
            accepted.add((a["task_id"], a["group_uid"], a["node_uid"]))
    sites = {}
    covered = set()
    for pair in pairs:
        task = pair["student_task_id"]
        node, _, turn = pair["node_uid"].rpartition(":t")
        site = pair["group_uid"]
        member = (task, pair["student_traj_uid"])
        if ((task, pair["original_group_uid"], node) not in accepted or
                task != pair["teacher_task_id"] or int(turn) != pair["divergence_turn"] or
                trajectories.get(task, {}).get(member[1]) != 0 or member in covered):
            raise ValueError("Local contrast has no matching verified recovery or repeats a student")
        covered.add(member)
        sites[site] = int(turn)
    if Counter(signals) != Counter(sites.keys()):
        raise ValueError("Actor divergence sites differ from the verified local contrasts")
    if any(site not in sites or step <= sites[site] for site, step in suffix):
        raise ValueError("Distillation example is not strictly after its verified divergence")
    return {
        "step": step, "group_size": 8, "tasks": len(trajectories), "student_trajectories": 64,
        "student_successes_per_task": [sum(v == 10 for v in trajectories[t].values())
                                       for t in sorted(trajectories)],
        "all_failure_groups": len(all_failed), "intervened_groups": len({t for t, _, _ in used}),
        "teacher_attempts": len(attempts), "root_attempts": sum(n for (_, _, root), n in used.items() if root),
        "verified_successful_attempts": sum(bool(a["success"]) for a in attempts),
        "divergence_sites_in_actor_batch": len(signals),
        "matched_student_trajectories": len(pairs), "strict_suffix_examples": len(suffix),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.step), indent=2))
