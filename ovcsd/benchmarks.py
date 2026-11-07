"""One serial environment worker. May run in a separate benchmark Python env."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys


class ALFWorld:
    def __init__(self, split, config):
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        with open(config) as fp:
            cfg = yaml.safe_load(fp)
        cfg["env"]["domain_randomization"] = False
        cfg["general"]["training_method"] = "dqn"  # TextWorld wrapper without expert plans
        cfg["rl"]["training"]["max_nb_steps_per_episode"] = 50
        self.factory = AlfredTWEnv(cfg, train_eval={"train": "train", "seen": "eval_in_distribution",
                                                    "test": "eval_out_of_distribution"}[split])
        key = {"train": "data_path", "seen": "eval_id_data_path", "test": "eval_ood_data_path"}[split]
        root = Path(os.path.expandvars(cfg["dataset"][key])).resolve()
        self.files = {Path(p).resolve().relative_to(root).as_posix(): p for p in self.factory.game_files}
        self.tasks = sorted(self.files)
        self.current = self.env = None

    @staticmethod
    def observation(text, info):
        return text[0].strip() + "\nAvailable actions: " + " | ".join(sorted(info["admissible_commands"][0]))

    def reset(self, task):
        if task != self.current:
            if self.env is not None:
                self.env.close()
            self.factory.game_files = [self.files[task]]
            self.env = self.factory.init_env(batch_size=1)
            self.current = task
        self.env.seed(0)
        obs, info = self.env.reset()
        if Path(info["extra.gamefile"][0]).resolve() != Path(self.files[task]).resolve():
            raise RuntimeError("ALFWorld reset returned the wrong game")
        goal = obs[0].split("Your task is to:")[-1].strip()
        return self.observation(obs, info), goal

    def step(self, action):
        obs, _, done, info = self.env.step([action])
        return self.observation(obs, info), float(info["won"][0]), bool(done[0])


class WebShop:
    def __init__(self, split, checkout):
        checkout = str(Path(checkout).resolve())
        sys.path.insert(0, checkout)
        os.chdir(checkout)  # upstream resolves search indices and data relative to the checkout
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        from web_agent_site.engine import engine

        # The upstream defaults point to the 1k product preview. Bind both full
        # catalog inputs explicitly, as in the benchmark's full-data setup.
        data = Path(checkout) / "data"
        engine.DEFAULT_ATTR_PATH = str(data / "items_ins_v2.json")
        self.env = WebAgentTextEnv(observation_mode="text", human_goals=True,
                                   file_path=str(data / "items_shuffle.json"))
        count = len(self.env.server.goals)
        self.tasks = list(map(str, {"train": range(1500, count), "seen": range(500, 1500),
                                   "test": range(500)}[split]))

    def observation(self, text):
        affordances = self.env.get_available_actions()
        actions = ["search[keywords]"] if affordances["has_search_bar"] else []
        actions.extend("click[" + item + "]" for item in affordances["clickables"])
        return text + "\nAvailable actions: " + " | ".join(actions)

    def reset(self, task):
        # Upstream keeps purchased items and goal metadata in per-session state.
        self.env.server.user_sessions.pop(str(task), None)
        obs, _ = self.env.reset(session=int(task))
        return self.observation(obs), self.env.get_instruction_text()

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        return self.observation(obs), float(reward), bool(done)


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["alfworld", "webshop"])
    parser.add_argument("split", choices=["train", "seen", "test"])
    parser.add_argument("resource")
    args = parser.parse_args()
    # Isolate protocol output from native libraries and nested subprocesses.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        random.seed(0)
        np.random.seed(0)
        backend = ALFWorld if args.benchmark == "alfworld" else WebShop
        env = backend(args.split, args.resource)
        for line in sys.stdin:
            request = json.loads(line)
            op = request["op"]
            if op == "tasks":
                value = env.tasks
            elif op == "reset":
                value = env.reset(request["arg"])
            elif op == "step":
                value = env.step(request["arg"])
            else:
                raise ValueError(op)
            print(json.dumps(value), file=output, flush=True)


if __name__ == "__main__":
    serve()
