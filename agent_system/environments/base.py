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

from typing import List, Tuple, Dict, Union, Any
import torch
import numpy as np
import os
from agent_system.environments.prompts import *
from collections import defaultdict

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        """
        Initialize the environment manager.
        
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - config: Configuration object.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.config = config
        self._skill_mask = None

    def _skill_attribution_cfg(self):
        try:
            skill_sd_cfg = self.config.get("skill_sd", {}) or {}
            return skill_sd_cfg.get("skill_attribution", {}) or {}
        except Exception:
            skill_sd_cfg = getattr(self.config, "skill_sd", {}) or {}
            return getattr(skill_sd_cfg, "skill_attribution", {}) or {}

    def _skill_attribution_enabled(self) -> bool:
        cfg = self._skill_attribution_cfg()
        try:
            return bool(cfg.get("enabled", False))
        except Exception:
            return bool(getattr(cfg, "enabled", False))

    def _skill_attribution_rng(self):
        rng = getattr(self, "_skill_attribution_rng_state", None)
        if rng is None:
            seed = int(self.config.env.get("seed", 0))
            rng = np.random.default_rng(seed)
            self._skill_attribution_rng_state = rng
        return rng

    def _reset_injected_skill_ids(self, batch_size: int) -> None:
        self.injected_skill_ids = (
            [[] for _ in range(batch_size)]
            if self._skill_attribution_enabled()
            else None
        )

    def _retrieve_task_memory(self, task_description: str, mem_config):
        top_k = int(mem_config.get('top_k', 10))
        if (
            self._skill_attribution_enabled()
            and hasattr(self.retrieval_memory, "retrieve_with_dropout")
        ):
            cfg = self._skill_attribution_cfg()
            memories, kept_ids = self.retrieval_memory.retrieve_with_dropout(
                task_description=task_description,
                top_k=top_k,
                dropout_rate=float(cfg.get("dropout_rate", 0.3)),
                rng=self._skill_attribution_rng(),
                skill_weights=getattr(self.retrieval_memory, "skill_weights", None),
            )
            return memories, list(kept_ids)

        memories = self.retrieval_memory.retrieve(
            task_description=task_description,
            top_k=top_k,
            similarity_threshold=mem_config.get('similarity_threshold', 0.7),
            max_tokens=mem_config.get('max_tokens', 2000),
            include_examples=mem_config.get('include_examples', False),
        )
        return memories, []

    def _attach_injected_skill_ids(self, infos) -> None:
        if not self._skill_attribution_enabled():
            return
        skill_ids = getattr(self, "injected_skill_ids", None)
        if skill_ids is None:
            return
        for index, info in enumerate(infos):
            info["skill_ids"] = list(skill_ids[index]) if index < len(skill_ids) else []

    def set_skill_mask(self, mask):
        """Configure which env rows receive retrieval/skill context.

        ``None`` preserves legacy behavior. Boolean arrays use True for the
        teacher/with-skill arm and False for the student/no-skill arm.
        """
        if mask is None:
            self._skill_mask = None
            return
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 1:
            raise ValueError(f"skill_mask must be 1-D, got shape {mask_arr.shape}")
        self._skill_mask = mask_arr

    def _is_teacher_arm(self, env_idx: int) -> bool:
        if self._skill_mask is None:
            return True
        if env_idx >= len(self._skill_mask):
            return bool(self._skill_mask[-1])
        return bool(self._skill_mask[env_idx])

    def _flipped_skill_mask(self):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            if self._skill_mask is None:
                yield False
                return
            saved = self._skill_mask
            try:
                self._skill_mask = ~saved
                yield True
            finally:
                self._skill_mask = saved

        return _ctx()

    def _legacy_alt_prompt_tensors_enabled(self) -> bool:
        cfg = getattr(self, "config", None)
        if cfg is None:
            return False
        try:
            skill_sd_cfg = cfg.get("skill_sd", {})
        except Exception:
            skill_sd_cfg = getattr(cfg, "skill_sd", {})
        if skill_sd_cfg is None:
            return False
        try:
            return bool(skill_sd_cfg.get("legacy_alt_prompt_tensors", False))
        except Exception:
            return bool(getattr(skill_sd_cfg, "legacy_alt_prompt_tensors", False))

    def _should_build_alt_text(self) -> bool:
        return self._skill_mask is not None and (
            self._legacy_alt_prompt_tensors_enabled()
            or bool(getattr(self, "_force_alt_text", False))
        )

    def _extract_failure_reflections(self, kwargs, batch_size: int):
        reflections = [None] * batch_size
        if kwargs is None:
            return reflections
        if isinstance(kwargs, dict):
            value = kwargs.get("skill_int_failure_reflection")
            return [value] * batch_size
        if isinstance(kwargs, (list, tuple, np.ndarray)):
            for i, item in enumerate(kwargs[:batch_size]):
                if isinstance(item, dict):
                    reflections[i] = item.get("skill_int_failure_reflection")
        return reflections

    def _failure_reflection_for_prompt(self, env_idx: int) -> str:
        reflections = getattr(self, "failure_reflections", None)
        if not reflections or not self._is_teacher_arm(env_idx):
            return ""
        try:
            value = reflections[env_idx]
        except Exception:
            return ""
        if value is None:
            return ""
        text = str(value).strip()
        return f"\n\n{text}" if text else ""

    def _should_use_retrieval_prompt(self, env_idx: int, init: bool = False) -> bool:
        has_retrieval = (
            getattr(self, "retrieval_memory", None) is not None
            and getattr(self, "retrieved_memories", None) is not None
        )
        if not has_retrieval or not self._is_teacher_arm(env_idx):
            return False
        memory_cfg = self.config.env.get("skills_only_memory", {})
        include_on_init = bool(memory_cfg.get("include_on_init", False))
        return (
            not init
            or include_on_init
            or bool(getattr(self, "_force_init_skill_prompt", False))
        )

    def reset(self, kwargs) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
        Parameters:
        - kwargs (Dict): Additional keyword arguments for resetting the environment.

        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos
    
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None, # Implement this if needed
            'image': next_obs,
            'anchor': None # For GiGPO only. anchor observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        
        return next_observations, rewards, dones, infos

    def build_text_obs(self,) -> List[str]:
        """
        This function builds the text observation for the agent.
        
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        pass

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return
            
    def save_image(self, image, step):
        """
        Save an image to a file.
        
        Parameters:
        - image (np.ndarray or torch.Tensor): The image to save.
        - path (str): The path to save the image.
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
        
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)
        
        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)
