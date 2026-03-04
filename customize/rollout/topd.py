from .agent_core_gen import generate as react_generate
from argparse import Namespace
from slime.utils.types import Sample
from slime.rollout.sglang_rollout import GenerateState
from typing import Any
import aiohttp
import torch
import asyncio
import logging
logger = logging.getLogger(__name__)

async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    sample = await react_generate(args, sample, sampling_params, evaluation)
    #only process completed sample
    if sample.status != Sample.Status.COMPLETED:
        return sample
    teacher_endpoint = args.rm_url
    state = GenerateState(args)
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async def call_teacher():
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{teacher_endpoint}/generate", json=payload) as resp:
                resp.raise_for_status()
                teacher_resp = await resp.json()
                return teacher_resp
    teacher_resp_task = asyncio.create_task(call_teacher())
    while not teacher_resp_task.done():
        if state.aborted:
            teacher_resp_task.cancel()
            sample.status = Sample.Status.ABORTED
            return sample
        await asyncio.sleep(1)
    try:
        teacher_resp = teacher_resp_task.result()
    except Exception as e:
        logger.warning(f"Error calling teacher: {e}")
        sample.status = Sample.Status.FAILED
        return sample
    teacher_log_probs = [i[0] for i in teacher_resp["meta_info"]["input_token_logprobs"]]
    #first token has no prob
    assert len(teacher_log_probs) == len(sample.tokens)
    assert teacher_log_probs[0] is None

    teacher_log_probs = teacher_log_probs[-sample.response_length:]

    sample.teacher_log_probs = torch.tensor(teacher_log_probs, dtype=torch.float32)
    return sample
