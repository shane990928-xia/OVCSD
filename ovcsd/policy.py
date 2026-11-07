"""Causal LM acting and action-span scoring under either prompt view."""
import re
from contextlib import nullcontext

import torch
from transformers import GenerationConfig

from .method import Action


def parse_action(tokenizer, ids):
    decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
    text = decode(ids)
    matches = list(re.finditer(r"<action>\s*(.+?)\s*</action>", text, re.I | re.S))
    if matches:
        begin, end = matches[-1].span(1)
    else:
        # Some action-only checkpoints emit a bare command. Malformed structured
        # replies consume one environment turn without supplying an action target.
        raw = text.strip()
        if not raw or "\n" in raw or "<" in raw or ">" in raw:
            return Action("invalid", tuple(ids), ())
        begin = text.index(raw)
        end = begin + len(raw)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(ids[:len(encoded.input_ids)]) == encoded.input_ids:
        offsets = encoded.offset_mapping
    else:
        # A sampled segmentation need not equal the tokenizer's preferred encoding.
        # Keep the original IDs (including any conditioning reasoning tokens).
        stops = [len(decode(ids[:i + 1])) for i in range(len(ids))]
        offsets = zip([0] + stops[:-1], stops)
    active = tuple(i for i, (lo, hi) in enumerate(offsets) if lo < end and hi > begin and hi > lo)
    return Action(text[begin:end], tuple(ids), active)


class Policy:
    def __init__(self, model, tokenizer, benchmark, prompt_limit, response_limit, temperature):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.benchmark = benchmark
        self.prompt_limit = prompt_limit
        self.response_limit = response_limit
        self.temperature = temperature

    def context(self, goal, steps, observation, skills):
        instruction = (f"Act in {self.benchmark}. Task: {goal}\n"
                       "Choose one environment command. Enclose it in <action> and </action>.")
        if skills is not None:
            instruction += "\nRetrieved skills:\n" + skills
        history = [f"Observation: {s.observation}\nAction: {s.action.command}" for s in steps]
        for start in range(len(history) + 1):
            content = "\n".join([instruction, *history[start:], "Observation: " + observation])
            ids = self.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=True, add_generation_prompt=True)
            if len(ids) <= self.prompt_limit:
                return ids
        # Extremely long individual observations: retain the instruction and the latest state.
        half = self.prompt_limit // 2
        return ids[:half] + ids[-(self.prompt_limit - half):]

    def precision(self):
        if self.model.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def act(self, goal, steps, observation, skills):
        prompt = self.context(goal, steps, observation, skills)
        tokens = torch.tensor([prompt], device=self.model.device)
        stop = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        generation = GenerationConfig(do_sample=True, temperature=self.temperature,
                                      top_k=0, top_p=1.0, max_new_tokens=self.response_limit,
                                      eos_token_id=stop, pad_token_id=self.tokenizer.eos_token_id)
        with self.precision():
            result = self.model.generate(tokens, attention_mask=torch.ones_like(tokens),
                                         generation_config=generation)
        return parse_action(self.tokenizer, result[0, len(prompt):].tolist())

    def logits(self, episode, depth, skills=None, model=None):
        step = episode.steps[depth]
        action = step.action
        prompt = self.context(episode.goal, episode.steps[:depth], step.observation, skills)
        model = self.model if model is None else model
        tokens = torch.tensor([prompt + list(action.response_ids)], device=model.device)
        # The first logit of this tail predicts the first generated token. Avoid
        # materializing prompt-length × vocabulary logits on long WebShop pages.
        with self.precision():
            tail = model(input_ids=tokens, use_cache=False,
                         logits_to_keep=len(action.response_ids) + 1).logits
            scores = tail[0, list(action.positions)]
        return scores.float()
