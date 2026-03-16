"""
End-to-end test for customize/rollout/rg.py generate function.

Prerequisites:
  - sglang server running at SGLANG_URL (default http://localhost:30000)
  - data_encode_server running at http://localhost:8100 (for reward calculation)
  - agent_core package installed

Usage:
  python -m customize.rollout.test_rg_e2e --hf-checkpoint <model_path>
  python -m customize.rollout.test_rg_e2e --hf-checkpoint <model_path> --sglang-url http://localhost:30000
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from urllib.parse import urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from slime.utils.types import Sample
from slime.utils.misc import SingletonMeta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_agent_core_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw["agent_core_config"]


def load_sample_query(jsonl_path: str, index: int = 0) -> str:
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == index:
                data = json.loads(line.strip())
                return data["query"]
    raise IndexError(f"Sample index {index} out of range")


def build_args(
    hf_checkpoint: str,
    sglang_url: str,
    rm_url: str,
    agent_core_config: dict,
    style_reward_factor: float = 1.0,
) -> Namespace:
    parsed = urlparse(sglang_url)
    host = parsed.hostname
    port = parsed.port

    return Namespace(
        hf_checkpoint=hf_checkpoint,
        sglang_router_ip=host,
        sglang_router_port=port,
        sglang_server_concurrency=512,
        sglang_dp_size=1,
        sglang_router_policy="round_robin",
        sglang_enable_deterministic_inference=False,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=0.6,
        rollout_top_p=0.95,
        rollout_top_k=-1,
        rollout_max_response_len=64000,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        rm_url=rm_url,
        style_reward_factor=style_reward_factor,
        use_tis=False,
        agent_core_config=agent_core_config,
    )


def build_sample(query: str, label: str = "unknown") -> Sample:
    return Sample(
        prompt=query,
        label=label,
        status=Sample.Status.PENDING,
    )


async def run_test(args: Namespace, sample: Sample, sampling_params: dict) -> Sample:
    from customize.rollout.rg import generate
    result = await generate(args, sample, sampling_params, evaluation=False)
    return result


def dump_debug_tokens(sample: Sample, args: Namespace):
    """Dump tokens and think-tag analysis to files for debugging."""
    if not sample.tokens:
        logger.info("No tokens to dump.")
        return

    import torch
    from slime.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    tokens_t = torch.tensor(sample.tokens)

    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    starts = torch.where(tokens_t == think_start_id)[0].tolist()
    ends = torch.where(tokens_t == think_end_id)[0].tolist()
    im_ends = torch.where(tokens_t == im_end_id)[0].tolist()

    print("\n" + "=" * 60)
    print("THINK TAG ANALYSIS")
    print(f"  <think>  (id={think_start_id}) count: {len(starts)}, positions: {starts}")
    print(f"  </think> (id={think_end_id}) count: {len(ends)}, positions: {ends}")
    print(f"  <|im_end|> (id={im_end_id}) count: {len(im_ends)}")
    print(f"  Total tokens: {len(sample.tokens)}")

    CONTEXT = 15
    all_positions = [(p, "<think>") for p in starts] + [(p, "</think>") for p in ends]
    all_positions.sort()
    for pos, tag in all_positions:
        lo = max(0, pos - CONTEXT)
        hi = min(len(sample.tokens), pos + CONTEXT + 1)
        snippet = tokenizer.decode(sample.tokens[lo:hi], skip_special_tokens=False)
        #print(f"\n  [{tag}] at pos {pos}:")
        #print(f"    ...{snippet}...")

    dump_path = PROJECT_ROOT / "debug_tokens.json"
    with open(dump_path, "w") as f:
        json.dump({
            "tokens": sample.tokens,
            "think_start_positions": starts,
            "think_end_positions": ends,
            "im_end_positions": im_ends,
            "response_length": sample.response_length,
            "loss_mask": sample.loss_mask,
            "token_count": len(sample.tokens),
            "traj": sample.metadata["traj"],
        }, f)
    logger.info("Token debug data saved to %s", dump_path)

    decoded_path = PROJECT_ROOT / "debug_decoded.txt"
    with open(decoded_path, "w") as f:
        f.write(tokenizer.decode(sample.tokens, skip_special_tokens=False))
    logger.info("Full decoded text saved to %s", decoded_path)
    print("=" * 60)


def print_result(sample: Sample):
    print("=" * 60)
    print(f"Status:            {sample.status}")
    print(f"Reward:            {sample.reward}")
    print(f"Response length:   {sample.response_length}")
    print(f"Token count:       {len(sample.tokens)}")
    if sample.loss_mask is not None:
        print(f"Loss mask sum:     {sum(sample.loss_mask)}")
    if sample.token_rewards is not None:
        import torch
        tr = sample.token_rewards if isinstance(sample.token_rewards, torch.Tensor) else torch.tensor(sample.token_rewards)
        print(f"Token rewards:     mean={tr.mean().item():.4f}, min={tr.min().item():.4f}, max={tr.max().item():.4f}")
    print(f"Custom metrics:    {sample.customized_metrics}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="E2E test for rg.py generate")
    parser.add_argument("--hf-checkpoint", type=str, default="ckpts/Qwen3.5-4B", help="HuggingFace model path for tokenizer")
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000", help="sglang server URL")
    parser.add_argument("--rm-url", type=str, default="http://localhost:30000", help="Reward model URL")
    parser.add_argument("--style-reward-factor", type=float, default=1.0)
    parser.add_argument("--sample-index", type=int, default=0, help="Index of sample in rg_example.jsonl")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(PROJECT_ROOT / "customize" / "configs" / "agent" / "rg.yaml"),
        help="Path to agent core config yaml",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "rg_example.jsonl"),
        help="Path to sample data jsonl",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()

    logger.info("Loading agent_core_config from %s", cli_args.config_path)
    agent_core_config = load_agent_core_config(cli_args.config_path)

    logger.info("Loading sample query (index=%d) from %s", cli_args.sample_index, cli_args.data_path)
    query = load_sample_query(cli_args.data_path, cli_args.sample_index)

    args = build_args(
        hf_checkpoint=cli_args.hf_checkpoint,
        sglang_url=cli_args.sglang_url,
        rm_url=cli_args.rm_url,
        agent_core_config=agent_core_config,
        style_reward_factor=cli_args.style_reward_factor,
    )

    sample = build_sample(query)

    sampling_params = {
        "temperature": 0.6,
        "max_new_tokens": 32000,
    }

    logger.info("sglang endpoint: http://%s:%s", args.sglang_router_ip, args.sglang_router_port)
    logger.info("rm endpoint:     %s", args.rm_url)
    logger.info("hf_checkpoint:   %s", args.hf_checkpoint)
    logger.info("Starting generate ...")

    # Clear singleton so GenerateState can be re-created with new args
    SingletonMeta.clear_instances(SingletonMeta)

    result = asyncio.run(run_test(args, sample, sampling_params))

    logger.info("Generate completed.")
    print_result(result)
    dump_debug_tokens(result, args)

    if result.status == Sample.Status.COMPLETED:
        logger.info("TEST PASSED: sample completed successfully")
    else:
        logger.warning("TEST FINISHED with status: %s", result.status)


if __name__ == "__main__":
    main()
