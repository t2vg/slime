from .agent_core_gen import generate as react_generate
from argparse import Namespace
from slime.utils.types import Sample
from slime.rollout.sglang_rollout import GenerateState
from typing import Any
import aiohttp
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

def load_teacher_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def tokenize_messages(tokenizer: PreTrainedTokenizer, messages: list[dict], tools: list[dict] = None):
    cur_token_ids = []
    assistant_masks = []
    assistant_spans = []
    prev_token_len = 0
    eos_id = tokenizer.eos_token_id
    for i in range(len(messages)):
        cur_msg = messages[i]
        if cur_msg['role'] == 'assistant':
            new_token_ids = tokenizer.apply_chat_template(messages[:i+1], tools=tools,add_generation_prompt=False)
            delta_token_ids = new_token_ids[prev_token_len:]
            if delta_token_ids[-1] != eos_id:
                delta_token_ids.append(eos_id)
            span_start = len(cur_token_ids)
            span_end = span_start + len(delta_token_ids)
            assistant_spans.append((span_start, span_end))
            cur_token_ids.extend(delta_token_ids)
            assistant_masks.extend([1] * len(delta_token_ids))
            prev_token_len = len(new_token_ids)
        else:
            add_generation_prompt = cur_msg['role'] in ['user', 'tool']
            new_token_ids = tokenizer.apply_chat_template(messages[:i+1], tools=tools,add_generation_prompt=add_generation_prompt)
            delta_token_ids = new_token_ids[prev_token_len:]
            cur_token_ids.extend(delta_token_ids)
            assistant_masks.extend([0] * len(delta_token_ids))
            prev_token_len = len(new_token_ids)
    return cur_token_ids, assistant_masks, assistant_spans



async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    sample = await react_generate(args, sample, sampling_params, evaluation)
    #only process completed sample
    if sample.status != Sample.Status.COMPLETED:
        return sample
    teacher_endpoint = args.rm_url
    traj = sample.metadata["traj"]
    messages = traj["messages"]
    tools = traj["tools"]
    state = GenerateState(args)
    if not hasattr(state, "teacher_tokenizer"):
        state.teacher_tokenizer = load_teacher_tokenizer(args.teacher_tokenizer_path)
    teacher_token_ids, teacher_assistant_masks, teacher_assistant_spans = tokenize_messages(state.teacher_tokenizer, messages, tools)
    payload = {
        "input_ids": teacher_token_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{teacher_endpoint}/generate", json=payload) as resp:
            resp.raise_for_status()
            teacher_resp = await resp.json()
    teacher_log_probs = [i[0] for i in teacher_resp["meta_info"]["input_token_logprobs"]]
    #first token has no prob
    assert len(teacher_log_probs) == len(teacher_token_ids)
    assert teacher_log_probs[0] is None

    teacher_turn_logp = [sum(teacher_log_probs[s:e]) / (e-s) for s,e in teacher_assistant_spans]
    #get spans of consecutive 1s in loss mask
    padded_mask = torch.tensor([0] + sample.loss_mask + [0])
    diff = padded_mask[1:] - padded_mask[:-1]
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    assert len(starts) == len(ends)
    assert len(starts) == len(teacher_turn_logp)

    aligned_teacher_logp = torch.zeros(len(sample.loss_mask))
    for start, end, logp in zip(starts, ends, teacher_turn_logp, strict=True):
        aligned_teacher_logp[start:end] = logp
    sample.teacher_log_probs = aligned_teacher_logp
    return sample
