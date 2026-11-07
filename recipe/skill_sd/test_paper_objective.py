import unittest
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from recipe.skill_sd.paper_objective import prepare_weights
from recipe.skill_sd.prefix_rescue import build_failure_prefix_tries, select_deepest_failure_frontier
from recipe.skill_sd.skill_sd_trainer import SkillSDTrainer
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.workers.actor.dp_actor import DataParallelPPOActor


class PaperObjectiveTest(unittest.TestCase):
    def test_actor_updates_from_locked_ray_batch_with_suffix(self):
        module = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
        parameter = module.weight
        before = parameter.detach().clone()
        n, length = 3, 2
        mask = torch.ones(n, length)
        zeros = torch.zeros_like(mask)
        frozen = torch.full((n, length, 1), -0.6931472)
        cache = {
            "skill_int_suffix_q_topk_log_probs": frozen,
            "skill_int_suffix_q_residual_mass": torch.full_like(mask, 0.5),
            "skill_int_suffix_q_topk_ids": torch.zeros(n, length, 1, dtype=torch.long),
        }
        tensors = dict(responses=zeros.long(), input_ids=zeros.long(), attention_mask=mask,
                       position_ids=zeros.long(), old_log_probs=zeros, ref_log_prob=zeros,
                       advantages=mask, paper_mask=mask, paper_pg_weight=torch.tensor([1., 1., 0.]),
                       paper_ref_weight=torch.ones(n), paper_suffix_weight=torch.tensor([0., 0., 1.]),
                       skill_int_is_rescue_suffix=torch.tensor([False, False, True]))
        batch = TensorDict(tensors, batch_size=[n]).lock_()

        def forward_suffix(**kwargs):
            logits = parameter.reshape(1, 1).expand(1, length)
            return torch.nn.functional.logsigmoid(logits).unsqueeze(-1), torch.sigmoid(-logits), None

        def step():
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            return norm

        actor = SimpleNamespace(
            actor_module=module, actor_optimizer=optimizer,
            config=SimpleNamespace(ppo_epochs=1, ppo_mini_batch_size=3, ppo_micro_batch_size_per_gpu=1,
                                   clip_ratio=0.2, kl_loss_type="low_var_kl", kl_loss_coef=0.01),
            _forward_micro_batch=lambda micro, temperature, calculate_entropy: (
                None, torch.nn.functional.logsigmoid(parameter).expand(len(micro), length)),
            _precompute_rescue_suffix_q_cache=lambda *args, **kwargs: cache,
            _forward_suffix_distribution=forward_suffix, _optimizer_step=step,
        )
        result = DataParallelPPOActor.update_policy(
            actor, SimpleNamespace(batch=batch, meta_info={"temperature": 1.0}))
        self.assertGreater(result["actor/grad_norm"][0], 0)
        self.assertFalse(torch.equal(parameter, before))
        self.assertTrue(batch.is_locked)
        self.assertNotIn("skill_int_suffix_q_topk_ids", batch.keys())

    def test_one_vote_per_trajectory(self):
        rewards = torch.tensor([[10.], [0.], [0.], [0.]])
        advantages, _ = compute_grpo_outcome_advantage(
            rewards, torch.ones_like(rewards), np.array(["x"] * 4),
            np.array(["a", "b", "b", "b"]), compute_mean_std_cross_steps=False,
        )
        torch.testing.assert_close(advantages[:, 0], torch.tensor([1., -1., -1., -1.]))

    def test_only_complete_all_failure_groups(self):
        owner = SimpleNamespace(config=SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=8))))
        stats = {name: SimpleNamespace(traj_count=n, success_frac=fraction) for name, n, fraction in
                 [("failure", 8, 0), ("mixed", 8, .125), ("success", 8, 1), ("incomplete", 7, 0)]}
        self.assertEqual(SkillSDTrainer._prefix_rescue_candidate_group_uids(owner, stats, list(stats)), ["failure"])

    def test_failure_trie_preserves_disjoint_frontiers(self):
        def frontier(rewards):
            rows = [(i, step) for i in range(len(rewards)) for step in (0, 1)]
            side = lambda i: "east" if i < 4 else "west"
            batch = SimpleNamespace(non_tensor_batch={
                "uid": ["task"] * len(rows), "traj_uid": [str(i) for i, _ in rows],
                "task_id": ["game"] * len(rows), "is_action_valid": [True] * len(rows),
                "step_idx": [s for _, s in rows], "episode_rewards": [rewards[i] for i, _ in rows],
                "observation": ["hall" if s == 0 else side(i) for i, s in rows],
                "model_response": [f"<action>{side(i) if s == 0 else 'look'}</action>" for i, s in rows],
                "next_observation": [side(i) for i, _ in rows],
            })
            tries, _ = build_failure_prefix_tries(batch, ["task"], success_threshold=10, max_depth=10)
            return [(j.prefix_actions, j.support) for j in select_deepest_failure_frontier(tries)]

        self.assertEqual(frontier([0] * 8), [(("east",), 4), (("west",), 4)])
        for invalid in ([0] * 7, [0] * 7 + [10], [0] * 7 + [float("nan")]):
            self.assertEqual(frontier(invalid), [])

    def test_suffix_preserves_sampled_token_ids_in_both_contexts(self):
        owner = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0), _int_value=lambda v, _: int(v))
        batch = SimpleNamespace(batch={"input_ids": torch.zeros(1, 8), "responses": torch.zeros(1, 4)},
                                meta_info={})
        example = dict(student_prompt_ids=[10, 11], teacher_prompt_ids=[12, 13, 14],
                       response_ids=[20, 0, 21], response="<action>look</action>", group_uid="task")
        data = SkillSDTrainer._build_prefix_rescue_suffix_proto(owner, batch, [example]).batch
        self.assertEqual(data["input_ids"].tolist(), [[0, 0, 10, 11, 20, 0, 21, 0]])
        self.assertEqual(data["skill_int_suffix_q_input_ids"].tolist(), [[0, 12, 13, 14, 20, 0, 21, 0]])
        self.assertEqual(data["response_mask"].tolist(), [[1, 1, 1, 0]])

    def test_duplicate_rows_and_nonempty_suffix_normalization(self):
        # Task x has two trajectories. Its second trajectory has a repeated padding row.
        # Tasks y and z each contribute a local site. Only the first has suffix tokens.
        n = 10
        tensor = torch.tensor
        mask = tensor([[1, 0], [1, 1], [1, 1], [1, 0], [1, 0],
                       [1, 0], [1, 1], [1, 1], [1, 1], [0, 0]], dtype=torch.bool)
        proto = DataProto.from_dict(
            tensors={
                "responses": torch.ones(n, 2, dtype=torch.long),
                "advantages": tensor([1, -1, -1, -1, -1, 1, 1, 0, 0, 0])[:, None] * mask,
                "skill_int_is_rescue_pg": tensor([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool),
                "skill_int_is_teacher_aux": tensor([0]*7 + [1, 1, 1], dtype=torch.bool),
                "skill_int_is_rescue_suffix": tensor([0]*7 + [1, 1, 0], dtype=torch.bool),
            },
            non_tensors={
                "uid": np.array(["x", "x", "x", "y", "z", "s1", "s2", "s1", "s1", "pad"], object),
                "traj_uid": np.array(["a", "b", "b", "c", "d", "e", "f", "s", "s", "pad"], object),
                "step_idx": np.array([0, 0, 0, 1, 2, 0, 0, 1, 1, 0]),
            },
        )
        prepare_weights(proto, lambda _: mask)

        def mass(key, indices):
            return (proto.batch[key][indices] * mask.sum(1)[indices] / n).sum().item()

        self.assertAlmostEqual(mass("paper_ref_weight", slice(None)), 1, places=6)
        self.assertAlmostEqual(mass("paper_pg_weight", slice(3, 7)), 1, places=6)
        self.assertAlmostEqual(mass("paper_suffix_weight", slice(None)), 1, places=6)
        self.assertAlmostEqual(mass("paper_ref_weight", slice(1, 3)), 1 / 6, places=6)


if __name__ == "__main__":
    unittest.main()
