"""Train or evaluate a skill-free checkpoint on ALFWorld or WebShop."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
from statistics import fmean
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .method import collect, rollout
from .objective import accumulate
from .policy import Policy
from .skills import fixed_memory


class Worker:
    def __init__(self, proc):
        self.proc = proc

    def call(self, op, arg=None):
        self.proc.stdin.write(json.dumps({"op": op, "arg": arg}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Benchmark worker exited ({self.proc.wait()}). See its stderr above.")
        return json.loads(line)

    def reset(self, task):
        return self.call("reset", task)

    def step(self, action):
        return self.call("step", action)


@contextmanager
def benchmark(args, split):
    script = str(Path(__file__).with_name("benchmarks.py").resolve())
    cmd = [args.env_python, script, args.benchmark, split, str(Path(args.resource).resolve())]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            encoding="utf-8", env=env)
    try:
        yield Worker(proc)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def task_ids(env, manifest):
    available = env.call("tasks")
    if manifest:
        selected = [line.strip() for line in Path(manifest).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        if len(set(selected)) != len(selected) or set(selected) - set(available):
            raise ValueError("Task manifest contains duplicates or IDs outside the selected split")
        available = selected
    if not available:
        raise ValueError("Empty benchmark split. Check the dataset installation and manifest.")
    return available


def train(args, policy, env, tasks, horizon, rng):
    if len(tasks) < 8:
        raise ValueError("Training requires at least eight distinct tasks")
    reference = load_model(args.model, args.device, getattr(policy.model, "hf_device_map", None))
    reference.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-6, weight_decay=.01, foreach=False)
    retrieve = fixed_memory(args.benchmark)
    for update in range(1, args.updates + 1):
        batch = rng.sample(tasks, 8)
        groups, sites, counts = collect(env, policy, batch, horizon,
                                        10 if args.benchmark == "alfworld" else horizon, retrieve)
        optimizer.zero_grad(set_to_none=True)
        stats = accumulate(policy, reference, groups, sites)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = dict(update=update, sites=len(sites), grad_norm=float(grad_norm),
                      student_success=fmean(ep.reward == 10 for g in groups for ep in g),
                      tasks=batch, **{k: counts[k] for k in ("student_actions", "replay_actions",
                                "teacher_actions", "teacher_attempts", "replay_mismatches")},
                      **{k: stats[k] for k in ("base", "ref_kl", "divergence", "weighted_suffix")})
        with (args.output / "training.jsonl").open("a", encoding="utf-8") as fp:
            print(json.dumps(record), file=fp)
        print(json.dumps({k: v for k, v in record.items() if k != "tasks"}), flush=True)
        if update % 25 == 0 or update == args.updates:
            dest = args.output / f"step-{update}"
            temp = dest.with_suffix(".partial")
            policy.model.save_pretrained(temp)
            policy.tokenizer.save_pretrained(temp)
            temp.rename(dest)


def evaluate(args, policy, env, tasks, horizon):
    wins, scores = [], []
    categories = defaultdict(list)
    with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as fp:
        for task in tasks:
            episode = rollout(env, policy, task, horizon, defaultdict(int))
            success = episode.reward == 10
            wins.append(success)
            scores.append(episode.score)
            if args.benchmark == "alfworld":
                categories[task.split("/")[0].split("-")[0]].append(success)
            print(json.dumps(dict(task=task, success=success, score=episode.score,
                                  actions=len(episode.steps))), file=fp, flush=True)
    metrics = dict(tasks=len(tasks), success_pct=100 * fmean(wins))
    if args.benchmark == "webshop":
        metrics["score_x100"] = 100 * fmean(scores)
    else:
        metrics["success_pct_by_category"] = {k: 100 * fmean(v) for k, v in categories.items()}
    print(json.dumps(metrics), flush=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_model(checkpoint, device, placement=None):
    if device == "balanced":
        if torch.cuda.device_count() < 2:
            raise ValueError("--device balanced requires at least two visible CUDA devices")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32,
                                                     device_map=placement or "balanced")
        if any(d in ("cpu", "disk") for d in model.hf_device_map.values()):
            raise ValueError("Insufficient GPU memory for the requested model placement")
        return model
    return AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("benchmark", choices=["alfworld", "webshop"])
    parser.add_argument("--model", required=True, help="HF-format supervised initialization or evaluation checkpoint")
    parser.add_argument("--resource", required=True, help="ALFWorld base_config.yaml or WebShop checkout")
    parser.add_argument("--env-python", default=sys.executable, help="Interpreter with the benchmark installed")
    parser.add_argument("--output", required=True, type=Path, help="New run directory")
    parser.add_argument("--tasks", help="Optional split-specific task manifest, one ID per line")
    parser.add_argument("--eval-split", choices=["test", "seen"], default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    training = args.mode == "train"
    with benchmark(args, "train" if training else args.eval_split) as env:
        tasks = task_ids(env, args.tasks)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = load_model(args.model, args.device)
        alf = args.benchmark == "alfworld"
        policy = Policy(model, tokenizer, args.benchmark, 4096 if alf else 6000,
                        512 if alf else 768, 1.0 if training else .4)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
        if training:
            train(args, policy, env, tasks, 50 if alf else 15, rng)
        else:
            evaluate(args, policy, env, tasks, 50 if alf else 15)


if __name__ == "__main__":
    main()
