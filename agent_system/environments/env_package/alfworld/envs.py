import fcntl
import shutil
# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =


def _cache_fast_downward_library():
    import fast_downward

    if getattr(fast_downward.load_lib, "_cpsd_single_env_cache", False):
        return

    cached_lib = None

    def load_persistent_lib():
        """Load libdownward from a persistent copy.

        fast_downward.interface.load_lib() extracts the shared object inside a
        TemporaryDirectory and removes it immediately after ``LoadLibrary``.
        On Python 3.12 this cleanup can race with the loader and raises
        ``OSError: [Errno 39] Directory not empty``.  Keeping the copied .so
        alive for the actor lifetime avoids both the cleanup race and loading
        an already-unlinked library.
        """
        import fast_downward.interface as fast_downward_interface

        cache_dir = os.environ.get(
            "CPSD_FAST_DOWNWARD_LIB_DIR", "/tmp/cpsd_fast_downward_libs"
        )
        os.makedirs(cache_dir, exist_ok=True)
        lib_path = os.path.join(cache_dir, f"libdownward-{os.getpid()}.so")
        if not os.path.isfile(lib_path):
            source = fast_downward_interface.DOWNWARD_LIB_PATH
            tmp_path = f"{lib_path}.tmp"
            shutil.copyfile(source, tmp_path)
            os.replace(tmp_path, lib_path)
        downward_lib = fast_downward_interface.cdll.LoadLibrary(lib_path)
        downward_lib.load_sas.argtypes = [fast_downward_interface.c_char_p]
        downward_lib.load_sas.restype = None
        downward_lib.load_sas_replan.argtypes = [fast_downward_interface.c_char_p]
        downward_lib.load_sas_replan.restype = None
        downward_lib.cleanup.argtypes = []
        downward_lib.cleanup.restype = None
        downward_lib.get_applicable_operators_count.argtypes = []
        downward_lib.get_applicable_operators_count.restype = fast_downward_interface.c_int
        downward_lib.get_applicable_operators.argtypes = [
            fast_downward_interface.POINTER(fast_downward_interface.Operator)
        ]
        downward_lib.get_applicable_operators.restype = None
        downward_lib.get_state_size.argtypes = []
        downward_lib.get_state_size.restype = fast_downward_interface.c_int
        downward_lib.get_state.argtypes = [fast_downward_interface.POINTER(fast_downward_interface.Atom)]
        downward_lib.get_state.restype = None
        downward_lib.apply_operator.argtypes = [
            fast_downward_interface.c_int,
            fast_downward_interface.POINTER(fast_downward_interface.Atom),
        ]
        downward_lib.apply_operator.restype = fast_downward_interface.c_int
        downward_lib.check_goal.argtypes = []
        downward_lib.check_goal.restype = fast_downward_interface.c_bool
        downward_lib.solve.argtypes = [fast_downward_interface.c_bool]
        downward_lib.solve.restype = fast_downward_interface.c_bool
        downward_lib.solve_sas.argtypes = [fast_downward_interface.c_char_p, fast_downward_interface.c_bool]
        downward_lib.solve_sas.restype = fast_downward_interface.c_bool
        downward_lib.replan.argtypes = [fast_downward_interface.c_bool]
        downward_lib.replan.restype = fast_downward_interface.c_bool
        downward_lib.get_last_plan_length.argtypes = []
        downward_lib.get_last_plan_length.restype = fast_downward_interface.c_int
        downward_lib.get_last_plan.argtypes = [
            fast_downward_interface.POINTER(fast_downward_interface.Operator)
        ]
        downward_lib.get_last_plan.restype = None
        downward_lib.check_solution.argtypes = [
            fast_downward_interface.c_int,
            fast_downward_interface.POINTER(fast_downward_interface.Operator),
        ]
        downward_lib.check_solution.restype = fast_downward_interface.c_bool
        return downward_lib

    def load_cached_lib():
        nonlocal cached_lib
        if cached_lib is None:
            # fast-downward extracts a shared library through tempfile. Ray
            # actors may inherit identical tempfile RNG state, so concurrent
            # first loads can target the same directory and race during cleanup.
            lock_path = os.environ.get(
                "CPSD_FAST_DOWNWARD_LOAD_LOCK",
                "/tmp/cpsd_fast_downward_load.lock",
            )
            with open(lock_path, "w", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                cached_lib = load_persistent_lib()
        else:
            cached_lib.cleanup()
        return cached_lib

    load_cached_lib._cpsd_single_env_cache = True
    fast_downward.load_lib = load_cached_lib

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.config = config
        self.seed = seed
        self.base_env = base_env
        self.default_game_files = list(base_env.game_files)
        self.default_num_games = base_env.num_games
        self.loaded_gamefile = None
        # libdownward cannot be unloaded because it contains GNU-unique symbols.
        # Each actor is serial and owns one batch-1 environment, so reuse one copy.
        if config["env"]["type"] == "AlfredTWEnv":
            _cache_fast_downward_library()
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)

    def _close_env(self):
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'close'):
            try:
                self.env.close()
            except Exception:
                pass

    def _rebuild_env(self, gamefile=None):
        self._close_env()
        if gamefile is None:
            self.base_env.game_files = list(self.default_game_files)
            self.base_env.num_games = self.default_num_games
            self.loaded_gamefile = None
        else:
            self.base_env.game_files = [gamefile]
            self.base_env.num_games = 1
            self.loaded_gamefile = gamefile
        self.env = self.base_env.init_env(batch_size=1)
        self.env.seed(self.seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self, gamefile=None):
        """Reset the environment"""
        if gamefile != self.loaded_gamefile:
            self._rebuild_env(gamefile)
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos

    def restart_game_sequence(self, gamefiles=None):
        """Restart a game iterator from this worker's original seed."""
        self._close_env()
        selected = self.default_game_files if gamefiles is None else list(gamefiles)
        if not selected:
            raise ValueError("Cannot restart an ALFWorld worker with no games")
        self.base_env.game_files = list(selected)
        self.base_env.num_games = len(selected)
        self.loaded_gamefile = None
        self.env = self.base_env.init_env(batch_size=1)
        self.env.seed(self.seed)
        return True
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.seed = int(seed)
        self.is_train = bool(is_train)
        self.default_game_files = list(base_env.game_files)
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.active_worker_indices = list(range(self.num_processes))
        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        active_indices = getattr(self, 'active_worker_indices', list(range(self.num_processes)))
        assert len(actions) == len(active_indices), \
            "The num of actions must be equal to the active num of processes"

        # Send step commands to all workers
        futures = []
        for action, worker_idx in zip(actions, active_indices):
            future = self.workers[worker_idx].step.remote(action)
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for worker_idx, (obs, scores, dones, info) in zip(active_indices, results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            if 'goal_condition_success_rate' in info:
                info['task_score'] = float(info['goal_condition_success_rate'])
            info_list.append(info)

            self.prev_admissible_commands[worker_idx] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self, gamefiles=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        if gamefiles is not None:
            if len(gamefiles) > self.num_processes:
                raise ValueError(f"Expected at most {self.num_processes} gamefiles, got {len(gamefiles)}")
            self.active_worker_indices = list(range(len(gamefiles)))
        else:
            self.active_worker_indices = list(range(self.num_processes))
        active_indices = self.active_worker_indices
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for local_idx, worker_idx in enumerate(active_indices):
            gamefile = None if gamefiles is None else gamefiles[local_idx]
            future = self.workers[worker_idx].reset.remote(gamefile)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for worker_idx, (obs, info) in zip(active_indices, results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[worker_idx] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def restart_game_sequences(self):
        """Reset every worker's game iterator without consuming a game."""
        gamefiles_by_worker = [None] * self.num_processes
        if not self.is_train:
            rng = np.random.RandomState(self.seed)
            order = rng.permutation(len(self.default_game_files))
            shuffled = [self.default_game_files[int(i)] for i in order]
            gamefiles_by_worker = [
                shuffled[worker_idx::self.num_processes]
                for worker_idx in range(self.num_processes)
            ]
        ray.get([
            worker.restart_game_sequence.remote(gamefiles_by_worker[i])
            for i, worker in enumerate(self.workers)
        ])
        self.active_worker_indices = list(range(self.num_processes))
        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        active_indices = getattr(self, 'active_worker_indices', list(range(self.num_processes)))
        return [self.prev_admissible_commands[i] for i in active_indices]

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)


if os.environ.get("OVCSD_ENV_PROCESS_BACKEND") == "1":
    from recipe.skill_sd.alfworld_process_backend import RayEnvironmentProxy
    ray = RayEnvironmentProxy(ray, AlfworldWorker)
