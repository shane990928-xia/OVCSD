"""Deterministic intervention tests and an offline CPU language-model update."""
import copy
import math
import unittest
from collections import Counter

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from ovcsd.method import Action, Site, Step, advantages, collect, first_difference, intervene
from ovcsd.objective import accumulate, clipped, coarsened_kl
from ovcsd.policy import Policy, parse_action
from ovcsd.skills import fixed_memory


class Corridor:
    def reset(self, task):
        self.state = "start"
        return self.state, task

    def step(self, action):
        self.state = {"walk": "fork", "right": "door", "finish": "won"}.get(action, "lost")
        return self.state, float(self.state == "won"), self.state in ("lost", "won")


class Script:
    def __init__(self):
        self.views = []

    def act(self, goal, steps, observation, skills):
        self.views.append(skills)
        command = {"start": "walk", "fork": "right" if skills else "left", "door": "finish"}[observation]
        return Action(command, ({"walk": 3, "left": 4, "right": 5, "finish": 6}[command],), (0,))


class MethodTests(unittest.TestCase):
    def test_two_phase_verified_grouping(self):
        agent = Script()
        groups, sites, counts = collect(Corridor(), agent, ["a", "b"], 4, 10, lambda _: "skill")
        self.assertEqual(agent.views[:32], [None] * 32)
        self.assertEqual(agent.views[32:], ["skill"] * 4)
        self.assertEqual([s.depth for s in sites], [1, 1])
        self.assertEqual([len(s.members) for s in sites], [8, 8])
        self.assertEqual([s.action.command for s in sites[0].suffix], ["finish"])
        self.assertEqual((counts["student_actions"], counts["replay_actions"], counts["teacher_actions"]),
                         (32, 2, 4))
        self.assertEqual(advantages(groups[0]), ([0.] * 8, True))
        groups[0][0].reward = 10.
        self.assertFalse(advantages(groups[0])[1])
        self.assertEqual(intervene(Corridor(), agent, groups[0], 4, 10, "skill", counts), [])

    def test_failed_replay_backs_off_to_root(self):
        groups, _, _ = collect(Corridor(), Script(), ["a"], 4, 10, lambda _: "skill")

        class Flaky(Corridor):
            first = True

            def step(self, action):
                obs, reward, done = super().step(action)
                if self.first:
                    self.first = False
                    obs = "unexpected replay observation"
                return obs, reward, done

        counts = Counter()
        sites = intervene(Flaky(), Script(), groups[0], 4, 10, "skill", counts)
        self.assertEqual((counts["replay_mismatches"], counts["teacher_actions"]), (1, 3))
        self.assertEqual(len(sites), 1)

    def test_failure_and_alignment_gates(self):
        groups, sites, _ = collect(Corridor(), Script(), ["a"], 4, 10, lambda _: "skill")
        self.assertEqual(intervene(Corridor(), Script(), groups[0], 4, 10, "", Counter()), [])
        branch = copy.deepcopy(sites[0].teacher)
        move = branch.steps[1]
        branch.steps[1] = Step("somewhere else", move.action, move.following)
        self.assertIsNone(first_difference(branch, groups[0][0], 1))
        # A correct action on the last turn still produces a local site.
        branch.steps = branch.steps[:2]
        self.assertEqual(Site(1, branch, groups[0], "skill").suffix, [])

    def test_numerics(self):
        old = torch.zeros(2)
        current = torch.log(torch.tensor([1.6, .5]))
        torch.testing.assert_close(clipped(current, old, 2), torch.tensor([-2.4, -1.]))
        torch.testing.assert_close(clipped(current, old, -2), torch.tensor([3.2, 1.6]))
        q = torch.tensor([[.6, .35, .05]], requires_grad=True)
        p = torch.tensor([[.4, .1, .5]], requires_grad=True)
        kl = coarsened_kl(q.log(), p.log(), 1)
        self.assertAlmostEqual(kl.item(), .6 * math.log(1.5) + .4 * math.log(2/3), places=6)
        kl.backward()
        self.assertIsNone(q.grad)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_site_weighting_and_strict_suffix(self):
        class Scores:
            def __init__(self):
                self.logits_by_action = torch.zeros(10, requires_grad=True)
                self.teacher_depths = []

            def logits(self, episode, depth, skills=None, model=None):
                word = episode.steps[depth].action.response_ids[0]
                scores = self.logits_by_action
                if model is not None:
                    return torch.zeros(1, 10)
                if skills is not None:
                    self.teacher_depths.append(depth)
                    return torch.arange(10.).unsqueeze(0)
                # Separate scalar per command permits inspecting the local PG gradient.
                return (torch.eye(10)[word] * scores).unsqueeze(0)

        groups, sites, _ = collect(Corridor(), Script(), ["goal"], 4, 10, lambda _: "skill", k=4)
        policy = Scores()
        two_sites = [sites[0], Site(1, sites[0].teacher, groups[0][:1], "skill")]
        accumulate(policy, object(), groups, two_sites)
        self.assertEqual(policy.teacher_depths, [2, 2])
        expected = .9 * ((2 / 5) + (1 / 2)) / 2
        self.assertAlmostEqual(policy.logits_by_action.grad[4].item(), expected, places=6)
        self.assertAlmostEqual(policy.logits_by_action.grad[5].item(), -expected, places=6)
        self.assertAlmostEqual(policy.logits_by_action.grad[3].item(), 0., places=6)

    def test_cpu_lm_update(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)
        words = ["[UNK]", "[EOS]", "skill", "walk", "left", "right", "finish", "goal", "fork"]
        backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
        tok.chat_template = "{{ messages[0]['content'] }}"
        cfg = GPT2Config(vocab_size=72, n_layer=1, n_head=1, n_embd=16, n_positions=128,
                         eos_token_id=1, pad_token_id=1)
        model = GPT2LMHeadModel(cfg)
        agent = Policy(model, tok, "alfworld", 100, 4, 1.)
        ref = copy.deepcopy(model).eval().requires_grad_(False)
        self.assertEqual(parse_action(tok, [3, 1]).positions, (0,))
        self.assertIsInstance(agent.act("goal", [], "fork", None), Action)
        groups, sites, _ = collect(Corridor(), Script(), ["goal"], 4, 10, lambda _: "skill", k=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        stats = accumulate(agent, ref, groups, sites)
        self.assertGreater(stats["weighted_suffix"], 0.)
        self.assertGreater(model.transformer.wte.weight.grad.norm().item(), 0.)
        self.assertTrue(all(p.grad is None for p in ref.parameters()))
        original = model.transformer.wte.weight.detach().clone()
        optimizer.step()
        self.assertTrue(torch.isfinite(model.transformer.wte.weight).all())
        self.assertFalse(torch.equal(original, model.transformer.wte.weight))

    def test_both_frozen_memories(self):
        for bench, goal in [("alfworld", "clean the plate"), ("webshop", "a red dress")]:
            self.assertTrue(fixed_memory(bench)(goal))


if __name__ == "__main__":
    unittest.main()
