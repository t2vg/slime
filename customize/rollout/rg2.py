from .agent_core_gen import generate as react_generate
from argparse import Namespace
from slime.utils.types import Sample
from slime.rollout.sglang_rollout import GenerateState
from typing import Any
import aiohttp
import torch
import asyncio
import logging
import json
import math
import uuid
logger = logging.getLogger(__name__)


def gaussian_length_penalty(length: int, mean: float = 150, window: float = 50, max_penalty: float = 5.0) -> float:
    """Gaussian-shaped penalty that is ~0 at `mean` and rapidly approaches
    `-max_penalty` when `|length - mean|` exceeds `window`.

    sigma is derived so that at exactly ±window the penalty already reaches
    ~95 % of max_penalty (window ≈ 2σ)."""
    sigma = window / 2.0
    return max_penalty * (math.exp(-((length - mean) ** 2) / (2 * sigma ** 2)) - 1)

ENCODE_SERVER_URL = "http://t2vg-2tfv5-server-0.t2vg-2tfv5:8100"

def encode_data(data: dict[str, Any]) -> list[int]:
    import requests
    resp = requests.post(f"{ENCODE_SERVER_URL}/encode_data", json=data)
    resp.raise_for_status()
    return resp.json()["input_ids"]

def rebuild_data_with_new_thoughts(sample: Sample) -> tuple[dict[str, Any], bool]:
    d = sample.metadata['task_result']
    query = json.loads(d.metadata.query)
    raw_traj = query['messages']
    new_thoughts = [m for m in d.traj if m.type == 'ai']
    idx = 0
    new_traj = []
    last_turn_empty = False
    for i,t in enumerate(raw_traj):
        if idx == len(new_thoughts):
            break
        if t['role'] == 'assistant':
            n = new_thoughts[idx]
            content = n.text.split("</think>")[0].split("<think>")[-1].strip()
            content = "<think>\n" + content + "\n</think>\n\n"
            if i == len(raw_traj) - 1:
                answer = t['content'].split("</think>")[-1].strip()
                if len(answer) == 0:
                    last_turn_empty = True
            else:
                answer = ""
            t['content'] = content + answer
            idx += 1
        new_traj.append(t)
    assert idx == len(new_thoughts)
    query['messages'] = new_traj
    return query, last_turn_empty

def get_consecutive_span(arr: list[int]) -> list[tuple[int, int]]:
    padded_mask = torch.tensor([0] + arr + [0])
    diff = padded_mask[1:] - padded_mask[:-1]
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    assert len(starts) == len(ends)
    return list(zip(starts, ends, strict=False))

def get_think_content_spans(tokens: torch.Tensor, think_start_id: int, think_end_id: int, im_end_id: int) -> list[tuple[int, int]]:
    """Pair each <think> with its next </think>.

    Raises ``ValueError`` when the spans are structurally invalid:
    * unequal number of <think> / </think>
    * <|im_end|> appears inside a span (cross-turn corruption)
    """
    starts = torch.where(tokens == think_start_id)[0]
    ends = torch.where(tokens == think_end_id)[0]
    if len(starts) != len(ends):
        raise ValueError(
            f"<think>/<think> count mismatch: {len(starts)} starts vs {len(ends)} ends"
        )
    im_ends = torch.where(tokens == im_end_id)[0]
    spans = []
    for s, e in zip(starts, ends, strict=True):
        if torch.any((im_ends > s) & (im_ends < e)):
            raise ValueError(
                f"<|im_end|> found inside think span [{int(s)}, {int(e)}]"
            )
        spans.append((int(s) + 1, int(e) + 1))
    return spans

def get_im_think_content_spans(
    tokens: torch.Tensor, im_start_id: int, im_end_id: int,
    assistant_id: int, role_prefix_len: int,
) -> list[tuple[int, int]]:
    """Extract thinking content spans from assistant turns.

    Only considers ``<|im_start|>`` tokens followed by the *assistant* role
    token, skipping system/user turns.  The role prefix (``assistant\\n``,
    whose token-length is *role_prefix_len*) is excluded from the returned
    span so that ``tokens[s:e]`` = pure thinking content + ``<|im_end|>``.
    """
    starts = torch.where(tokens == im_start_id)[0]
    ends = torch.where(tokens == im_end_id)[0]
    spans = []
    for s in starts:
        if tokens[int(s) + 1] != assistant_id:
            continue
        candidates = ends[ends > s]
        if len(candidates) == 0:
            raise ValueError(
                f"No <|im_end|> found after <|im_start|> at position {int(s)}"
            )
        e = candidates[0]
        inner_starts = starts[(starts > s) & (starts < e)]
        if len(inner_starts) > 0:
            raise ValueError(
                f"Nested <|im_start|> inside thinking span [{int(s)}, {int(e)}]"
            )
        content_start = int(s) + 1 + role_prefix_len
        spans.append((content_start, int(e) + 1))
    return spans

def get_tokens_with_new_thoughts(
    sample: Sample, think_start_id: int, think_end_id: int, im_end_id: int, im_start_id: int,
    assistant_id: int, role_prefix_len: int,
) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], bool]:
    generated_tokens = torch.tensor(sample.tokens)
    data, last_turn_invalid = rebuild_data_with_new_thoughts(sample)
    tokens_with_new_thoughts = encode_data(data)
    tokens_with_new_thoughts = torch.tensor(tokens_with_new_thoughts)

    gen_spans = get_im_think_content_spans(generated_tokens, im_start_id, im_end_id, assistant_id, role_prefix_len)
    new_spans = get_think_content_spans(tokens_with_new_thoughts, think_start_id, think_end_id, im_end_id)

    if len(gen_spans) != len(new_spans):
        raise ValueError(
            f"Reasoning span count mismatch: generated={len(gen_spans)} vs rebuilt={len(new_spans)}"
        )

    result = tokens_with_new_thoughts.tolist()
    result_spans = []
    offset = 0
    for (gs, ge), (ns, ne) in zip(gen_spans, new_spans, strict=True):
        gen_content = generated_tokens[gs:ge - 1].tolist()
        adj_ns = int(ns) + offset
        adj_ne = int(ne) + offset
        result[adj_ns:adj_ne - 1] = gen_content
        new_len = int(ge) - int(gs)
        result_spans.append((adj_ns, adj_ns + new_len))
        offset += new_len - (int(ne) - int(ns))

    result_tensor = torch.tensor(result)
    think_end_positions = torch.where(result_tensor == think_end_id)[0]
    im_end_positions = torch.where(result_tensor == im_end_id)[0]
    im_start_positions = torch.where(result_tensor == im_start_id)[0]
    action_spans = []
    for te_pos in think_end_positions:
        candidates = im_end_positions[im_end_positions > te_pos]
        if len(candidates) == 0:
            raise ValueError(f"No <|im_end|> found after </think> at position {int(te_pos)}")
        ie_pos = candidates[0]
        if torch.any((im_start_positions > te_pos) & (im_start_positions < ie_pos)):
            raise ValueError(
                f"<|im_start|> found between </think>@{int(te_pos)} and <|im_end|>@{int(ie_pos)}, "
                "action span crosses turn boundary"
            )
        action_spans.append((int(te_pos) + 1, int(ie_pos) + 1))

    return result, result_spans, action_spans, gen_spans, last_turn_invalid


def build_baseline_tokens_and_spans(
    rebuild_tokens: list[int], think_spans: list[tuple[int, int]],
    think_end_id: int, im_end_id: int, im_start_id: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Build a token sequence with empty thinking content for baseline logp.

    For each think span, all content tokens are removed and only the
    ``</think>`` delimiter is kept, yielding ``<think></think>`` structure.
    Returns the new token list and the corresponding action spans.
    """
    result = list(rebuild_tokens)
    offset = 0
    for ts, te in think_spans:
        adj_ts = ts + offset
        adj_te = te + offset
        result[adj_ts:adj_te] = [think_end_id]
        offset -= (te - ts - 1)

    result_tensor = torch.tensor(result)
    think_end_positions = torch.where(result_tensor == think_end_id)[0]
    im_end_positions = torch.where(result_tensor == im_end_id)[0]
    im_start_positions = torch.where(result_tensor == im_start_id)[0]

    baseline_action_spans = []
    for te_pos in think_end_positions:
        candidates = im_end_positions[im_end_positions > te_pos]
        if len(candidates) == 0:
            raise ValueError(f"No <|im_end|> after </think> at {int(te_pos)} in baseline")
        ie_pos = candidates[0]
        if torch.any((im_start_positions > te_pos) & (im_start_positions < ie_pos)):
            raise ValueError(
                f"<|im_start|> between </think>@{int(te_pos)} and <|im_end|>@{int(ie_pos)} in baseline"
            )
        baseline_action_spans.append((int(te_pos) + 1, int(ie_pos) + 1))

    return result, baseline_action_spans


async def _fetch_rm_logprobs(
    session: aiohttp.ClientSession, rm_endpoint: str, tokens: list[int],
) -> tuple[list[float | None], list[float | None]]:
    """Fetch log-probs and entropy from the reward model for a token sequence."""
    payload = {
        "input_ids": tokens,
        "sampling_params": {"temperature": 1, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_entropy": True,
    }
    async with session.post(f"{rm_endpoint}/generate", json=payload) as resp:
        resp.raise_for_status()
        rm_resp = await resp.json()
    log_probs = [i[0] for i in rm_resp["meta_info"]["input_token_logprobs"]]
    entropy = rm_resp["meta_info"]["input_token_entropy"]
    assert len(log_probs) == len(tokens)
    assert len(entropy) == len(tokens)
    assert log_probs[0] is None
    assert entropy[0] is None
    return log_probs, entropy


async def calculate_turn_reward(
    args: Namespace, sample: Sample, think_start_id: int, think_end_id: int,
    im_end_id: int, im_start_id: int, assistant_id: int, role_prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rm_endpoint = args.rm_url
    sglang_endpoint = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    rebuild_tokens, think_spans, action_spans, sample_think_spans, last_turn_invalid = get_tokens_with_new_thoughts(sample, think_start_id, think_end_id, im_end_id, im_start_id, assistant_id, role_prefix_len)
    baseline_tokens, baseline_action_spans = build_baseline_tokens_and_spans(
        rebuild_tokens, think_spans, think_end_id, im_end_id, im_start_id
    )
    if len(baseline_action_spans) != len(action_spans):
        with open(f"/data/gongrui/slime_tmp/mismatch_{uuid.uuid4()}.json", "w") as f:
            json.dump({
                "rebuild_tokens": rebuild_tokens,
                "think_spans": think_spans,
                "action_spans": action_spans,
                "baseline_tokens": baseline_tokens,
                "baseline_action_spans": baseline_action_spans,
            }, f, indent=2)
        raise ValueError(
            f"Action span count mismatch: rebuild={len(action_spans)} vs baseline={len(baseline_action_spans)}; details dumped to mismatch_*.json"
        )

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        (rm_log_probs, rm_entropy), (baseline_log_probs, _) = await asyncio.gather(
            _fetch_rm_logprobs(session, rm_endpoint, rebuild_tokens),
            _fetch_rm_logprobs(session, rm_endpoint, baseline_tokens),
        )

    for (s, e), (bs, be) in zip(action_spans, baseline_action_spans):
        assert rebuild_tokens[s:e] == baseline_tokens[bs:be], \
            f"Action token mismatch: rebuild[{s}:{e}]={rebuild_tokens[s:e]} != baseline[{bs}:{be}]={baseline_tokens[bs:be]}"
    turn_rewards = [sum(rm_log_probs[s:e])/(e-s) if e-s > 1 else None for s, e in action_spans]
    baseline_rewards = [sum(baseline_log_probs[s:e])/(e-s) if e-s > 1 else None for s, e in baseline_action_spans]
    info_gain = [
        (t - b if t is not None and b is not None else None)
        for t, b in zip(turn_rewards, baseline_rewards)
    ]
    assert len(sample.metadata["output_token_mask"]) == len(sample.tokens)
    #turn_span = get_consecutive_span(sample.metadata["output_token_mask"])
    assert len(sample_think_spans) == len(info_gain)
    reasonable_rewards = torch.zeros(len(sample.tokens))
    length_penalties: list[float] = []
    for i, (s, e) in enumerate(sample_think_spans):
        if info_gain[i] is None:
            # Mask invalid turn tokens
            sample.metadata["output_token_mask"][s:e] = 0
            continue
        if i == len(sample_think_spans) - 1 and last_turn_invalid:
            # Mask invalid last turn tokens
            sample.metadata["output_token_mask"][s:e] = 0
            continue
        # e-1 is the im_end position
        assert sample.tokens[e-1] == im_end_id
        assert args.reasonable_temperature is not None
        reasonable_rewards[e-1] = math.tanh(info_gain[i]/args.reasonable_temperature)
        lp = gaussian_length_penalty(e - s, mean=150, window=50, max_penalty=0)
        reasonable_rewards[e-1] += lp
        length_penalties.append(lp)
    #reassign loss mask
    sample.loss_mask = sample.metadata["output_token_mask"][-sample.response_length:]

    sample_tokens = torch.tensor(sample.tokens)
    rebuild_tokens_tensor = torch.tensor(rebuild_tokens)

    style_reward = torch.zeros(len(sample.tokens))
    for (ts, te), (rs, re) in zip(sample_think_spans, think_spans, strict=True):
        assert (sample_tokens[ts:te - 1] == rebuild_tokens_tensor[rs:re - 1]).all(), \
            f"Token mismatch: sample[{ts}:{te-1}]={sample_tokens[ts:te-1].tolist()} != rebuild[{rs}:{re-1}]={rebuild_tokens_tensor[rs:re-1].tolist()}"
        phi = torch.tensor(rm_log_probs[rs:re]) + torch.tensor(rm_entropy[rs:re])
        #style_reward[ts:te] += (torch.sigmoid(phi/10) - 0.5) * 2
        avg_phi = phi.mean()
        assert args.style_temperature is not None
        style_reward[te-1] = torch.tanh(avg_phi/args.style_temperature).clip(-1, 0.1)

    valid_gains = [g for g in info_gain if g is not None]
    if valid_gains:
        sample.customized_metrics["avg_info_gain"] = sum(valid_gains) / len(valid_gains)
        sample.customized_metrics["max_info_gain"] = max(valid_gains)
        sample.customized_metrics["min_info_gain"] = min(valid_gains)
    if length_penalties:
        sample.customized_metrics["avg_length_penalty"] = sum(length_penalties) / len(length_penalties)
        sample.customized_metrics["max_length_penalty"] = max(length_penalties)
        sample.customized_metrics["min_length_penalty"] = min(length_penalties)

    reasonable_rewards = reasonable_rewards[-sample.response_length:]
    style_reward = style_reward[-sample.response_length:]
    return reasonable_rewards, style_reward


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    sample = await react_generate(args, sample, sampling_params, evaluation)
    #only process completed sample
    if sample.status != Sample.Status.COMPLETED:
        return sample
    state = GenerateState(args)
    tokenizer = state.tokenizer
    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert think_start_id is not None
    assert think_end_id is not None
    assert im_end_id is not None
    assert im_start_id is not None
    role_prefix_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
    assistant_id = role_prefix_tokens[0]
    role_prefix_len = len(role_prefix_tokens)

    calculate_reward_task = asyncio.create_task(calculate_turn_reward(args, sample, think_start_id, think_end_id, im_end_id, im_start_id, assistant_id, role_prefix_len))
    while not calculate_reward_task.done():
        if state.aborted:
            calculate_reward_task.cancel()
            sample.status = Sample.Status.ABORTED
            return sample
        await asyncio.sleep(1)
    try:
        reasonable_rewards, style_reward = calculate_reward_task.result()
    except Exception as e:
        import traceback
        logger.warning(f"Error calculating turn reward: {e}\n{traceback.format_exc()}")
        sample.status = Sample.Status.FAILED
        return sample
    token_rewards = torch.stack([reasonable_rewards, style_reward], dim=-1)  # [resp_len, 2]
    #loss_mask = torch.tensor(sample.loss_mask, dtype=torch.float32)
    #loss_mask[style_reward == 0] = 0
    #sample.loss_mask = loss_mask.tolist()
    rw = args.reasonable_reward_weight
    sw = 1 - rw
    avg_reasonable_rewards = reasonable_rewards.sum() / reasonable_rewards.nonzero().numel()
    avg_style_reward = style_reward.sum() / style_reward.nonzero().numel()
    sample.customized_metrics["avg_reasonable_reward"] = avg_reasonable_rewards.item()
    sample.customized_metrics["avg_style_reward"] = avg_style_reward.item()
    sample.customized_metrics["avg_token_reward"] = (avg_reasonable_rewards * rw + avg_style_reward * sw).item()
    sample.customized_metrics["max_reasonable_reward"] = reasonable_rewards.max().item()
    sample.customized_metrics["max_style_reward"] = style_reward.max().item()
    sample.customized_metrics["min_reasonable_reward"] = reasonable_rewards[reasonable_rewards.nonzero()].min().item()
    sample.customized_metrics["min_style_reward"] = style_reward.min().item()
    sample.token_rewards = token_rewards
    
    return sample


if __name__ == "__main__":
    from unittest.mock import patch, MagicMock

    S, E, IM, IMS = 1, 2, 3, 4  # think_start_id, think_end_id, im_end_id, im_start_id
    A, NL = 5, 6               # assistant token, newline token
    RPL = 2                     # role_prefix_len: "assistant\n" = 2 tokens (A + NL)

    def run_test(generated, new_thoughts, expected_result, expected_contents, expected_actions, label):
        mock_sample = MagicMock()
        mock_sample.tokens = generated
        with patch(f"{__name__}.encode_data", return_value=new_thoughts), \
             patch(f"{__name__}.rebuild_data_with_new_thoughts", return_value=({}, False)):
            result, spans, action_spans, _gen_spans, _invalid = get_tokens_with_new_thoughts(mock_sample, S, E, IM, IMS, A, RPL)
        assert result == expected_result, f"{label}: result {result} != expected {expected_result}"
        for i, (content, (s, e)) in enumerate(zip(expected_contents, spans, strict=True)):
            assert result[s:e] == content, \
                f"{label}: span {i} content {result[s:e]} != expected {content}"
        assert action_spans == expected_actions, \
            f"{label}: action_spans {action_spans} != expected {expected_actions}"
        for s, e in action_spans:
            assert result[s:e] == expected_result[s:e], \
                f"{label}: action content mismatch at ({s},{e})"
        print(f"{label}: PASSED")

    # 1: 两段 reasoning，长度不同；每段 IMS 后有 A NL role prefix
    run_test(
        generated=   [10, IMS, A, NL, 100, 101, 102, IM, 20, 21, IM, IMS, A, NL, 200, 201, IM, 30, IM],
        new_thoughts=[77, S, 300, 301, E, 88, 89, IM, S, 400, 401, 402, 403, E, 99, IM],
        expected_result=[77, S, 100, 101, 102, E, 88, 89, IM, S, 200, 201, E, 99, IM],
        expected_contents=[[100, 101, 102, E], [200, 201, E]],
        expected_actions=[(6, 9), (13, 15)],
        label="test_multi_span_diff_len",
    )

    # 2: 单段 reasoning，等长替换；think_end 后紧跟 im_end（action 为空 span）
    run_test(
        generated=   [10, IMS, A, NL, 100, 101, IM, IM, 20],
        new_thoughts=[77, S, 300, 301, E, IM, 88],
        expected_result=[77, S, 100, 101, E, IM, 88],
        expected_contents=[[100, 101, E]],
        expected_actions=[(5, 6)],
        label="test_single_span_empty_action",
    )

    # 3: 单段 reasoning，generated 更短；有 action token
    run_test(
        generated=   [10, IMS, A, NL, 100, IM, 20, 21, 22, IM],
        new_thoughts=[77, S, 300, 301, 302, E, 88, 89, IM],
        expected_result=[77, S, 100, E, 88, 89, IM],
        expected_contents=[[100, E]],
        expected_actions=[(4, 7)],
        label="test_single_span_shorter",
    )

    # 4: 三段 reasoning；每段后有 action
    run_test(
        generated=   [IMS, A, NL, 10, 11, IM, 50, IM, IMS, A, NL, 20, IM, 60, 61, IM, IMS, A, NL, 30, 31, 32, IM, 70, IM],
        new_thoughts=[S, 40, E, 55, IM, S, 50, 51, E, 65, IM, S, 60, E, 75, 76, IM],
        expected_result=[S, 10, 11, E, 55, IM, S, 20, E, 65, IM, S, 30, 31, 32, E, 75, 76, IM],
        expected_contents=[[10, 11, E], [20, E], [30, 31, 32, E]],
        expected_actions=[(4, 6), (9, 11), (16, 19)],
        label="test_three_spans",
    )

    # 5: 含 system/user 轮的 im_start 应被过滤，只取 assistant 轮
    SYS, USR = 7, 8  # system/user role tokens
    run_test(
        generated=   [IMS, SYS, 77, 78, IM, IMS, USR, 88, IM, IMS, A, NL, 100, 101, IM, 20, IM],
        new_thoughts=[77, S, 300, 301, E, 88, 89, IM],
        expected_result=[77, S, 100, 101, E, 88, 89, IM],
        expected_contents=[[100, 101, E]],
        expected_actions=[(5, 8)],
        label="test_filter_non_assistant",
    )

    # Test build_baseline_tokens_and_spans
    rebuild = [77, S, 100, 101, 102, E, 88, 89, IM, S, 200, 201, E, 99, IM]
    think_spans = [(2, 6), (10, 13)]
    baseline, b_spans = build_baseline_tokens_and_spans(rebuild, think_spans, E, IM, IMS)
    assert baseline == [77, S, E, 88, 89, IM, S, E, 99, IM], f"baseline: {baseline}"
    assert b_spans == [(3, 6), (8, 10)], f"baseline action spans: {b_spans}"
    for (oas, oae), (bas, bae) in zip([(6, 9), (13, 15)], b_spans):
        assert rebuild[oas:oae] == baseline[bas:bae], "action content mismatch"
    print("test_baseline: PASSED")

    # Single span: empty action after </think> (action span has 1 token = im_end only)
    rebuild2 = [77, S, 100, 101, E, IM, 88]
    think_spans2 = [(2, 5)]
    baseline2, b_spans2 = build_baseline_tokens_and_spans(rebuild2, think_spans2, E, IM, IMS)
    assert baseline2 == [77, S, E, IM, 88], f"baseline2: {baseline2}"
    assert b_spans2 == [(3, 4)], f"baseline2 action spans: {b_spans2}"
    print("test_baseline_single_empty_action: PASSED")

    print("All tests passed.")