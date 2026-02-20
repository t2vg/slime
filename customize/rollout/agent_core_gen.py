from argparse import Namespace
import logging
from slime.utils.types import Sample
from slime.rollout.sglang_rollout import GenerateState
from typing import Any
from agent_core.runner import execute_task
from agent_core.utils import clear_task_cache
from agent_core.protocol import TaskInput, FinishReason, AIMessage
from agent_core.config import set_error_info_depth, set_sqlite_path
import aiohttp
import asyncio
from copy import deepcopy

set_error_info_depth(1)

logger = logging.getLogger(__name__)

set_sqlite_path("/tmp/agent_core_session.sqlite")

async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    state = GenerateState(args)

    if sample.status == Sample.Status.COMPLETED:
        return sample

    config = deepcopy(args.agent_core_config)
    assert config['workflow']['name'] == "react_ctx_lim"
    sglang_endpoint = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    assert isinstance(sample.prompt, str), "Prompt should be a string"
    assert sample.label is not None, "Label should not be None"
    assert (
        sample.status in [Sample.Status.PENDING, Sample.Status.ABORTED, Sample.Status.FAILED]
    ), f"Sample status is {sample.status}"


    query = sample.prompt

    config["query"] = query
    
    task_input: TaskInput = sample.metadata.get("task_input", TaskInput.model_validate(config))
    task_input.endpoint_cfg.base_url = [f"{sglang_endpoint}/v1"]
    task_input.endpoint_cfg.mode = "response"
    task_input.workflow.workflow_args['verify'] = True
    task_input.ground_truth = sample.label
    task_input.model_cfg.max_context_length = sampling_params["max_new_tokens"]
    task_input.model_cfg.temperature = sampling_params["temperature"]

    sample.metadata['task_input'] = task_input

    if state.aborted:
        sample.status = Sample.Status.ABORTED
        return sample

    task = asyncio.create_task(execute_task(task_input,keep_session=True))

    async def trigger_abort():
        while not state.aborted:
            await asyncio.sleep(0.5)
        task.cancel()
    
    abort_trigger = asyncio.create_task(trigger_abort())

    result = await task
    abort_trigger.cancel()


    async def clear_cache():
        traj = result.traj
        first_ai_message = next(msg for msg in traj if isinstance(msg, AIMessage))
        #sglang removes all chained cache after the first AI message
        root_id = first_ai_message.response_metadata["id"]
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{sglang_endpoint}/trajectory/{root_id}") as response:
                response.raise_for_status()
        await clear_task_cache(task_input)

    if result.metadata.finish_reason not in [FinishReason.COMPLETED, FinishReason.ABORTED]:
        logger.warning(f"Rollout error: {result.metadata.error_info}")

    if result.metadata.finish_reason in [FinishReason.COMPLETED, FinishReason.INVALID_TOOL_ARGS, FinishReason.FAILED_TOOL_CALL]:
        last_message = result.traj[-1]
        assert isinstance(last_message, AIMessage), "Last message should be an AI message"
        traj_id = last_message.response_metadata["id"]
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{sglang_endpoint}/trajectory/{traj_id}") as response:
                traj = await response.json()
        token_ids:list[int] = traj["token_ids"]
        output_token_mask:list[int] = traj["output_token_mask"]
        assert len(token_ids) == len(output_token_mask), "Token ids and output token mask should have the same length"
        reward = result.metadata.metrics.get("score", 0.0)

        sample.tokens = token_ids
        sample.response = state.tokenizer.decode(token_ids, skip_special_tokens=False)
        #sample.response = result.traj
        try:
            first_response_idx = output_token_mask.index(1)
        except ValueError:
            first_response_idx = None
        
        if first_response_idx is None:
            logger.warning(f"No response tokens found in trajectory {traj_id}, all masks are zero")
            sample.loss_mask = []
            sample.response_length = 0
        else:
            sample.response_length = len(token_ids) - first_response_idx
            sample.loss_mask = output_token_mask[first_response_idx:]
        
        assert len(sample.loss_mask) == sample.response_length, \
            f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}"
        
        sample.reward = reward
        sample.status = Sample.Status.COMPLETED
        #clean cache
        await clear_cache()
    elif result.metadata.finish_reason in [FinishReason.ABORTED, FinishReason.INTERNAL_ERROR]:
        sample.status = Sample.Status.ABORTED
        #Only when state.aborted is True, the sample will be collected into the data buffer
        #This makes sure failed samples are collected into the data buffer.
        while not state.aborted:
            await asyncio.sleep(1)

    return sample