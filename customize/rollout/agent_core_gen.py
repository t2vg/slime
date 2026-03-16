from argparse import Namespace
import logging
from slime.utils.types import Sample
from slime.rollout.sglang_rollout import GenerateState
from typing import Any
from agent_core.runner import execute_task
from agent_core.utils import clear_task_cache
from agent_core.protocol import TaskInput, FinishReason, AIMessage
from agent_core.config import set_error_info_depth, set_sqlite_path
import requests
import asyncio
import traceback
from copy import deepcopy
import torch

set_error_info_depth(1)

logger = logging.getLogger(__name__)

set_sqlite_path("/tmp/agent_core_session.sqlite")


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    state = GenerateState(args)

    if sample.status == Sample.Status.COMPLETED:
        return sample

    if sample.status == Sample.Status.ABORTED and sample.tokens:
        sample.status = Sample.Status.COMPLETED
        return sample
    
    if sample.status == Sample.Status.FAILED:
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

    try:
        result = await asyncio.wait_for(task, timeout=3600)
    except Exception as e:
        logger.warning(f"Unexpected error in rollout: {e}")
        sample.status = Sample.Status.FAILED
        return sample
    finally:
        abort_trigger.cancel()


    async def clear_cache():
        traj = result.traj
        first_ai_message = next(msg for msg in traj if isinstance(msg, AIMessage))
        #sglang removes all chained cache after the first AI message
        root_id = first_ai_message.response_metadata["id"]
        response = requests.delete(f"{sglang_endpoint}/trajectory/{root_id}", timeout=60)
        if not response.ok:
            logger.warning(f"Failed to clear traj cache: {response.text}")
        try:
            await asyncio.wait_for(clear_task_cache(task_input), timeout=1800)
        except Exception as e:
            logger.warning(f"Failed to clear task cache: {traceback.format_exc()}")

    if result.metadata.finish_reason not in [FinishReason.COMPLETED, FinishReason.ABORTED]:
        logger.warning(f"Rollout error: {result.metadata.error_info}")
    
    valid_finish_reasons = [FinishReason.COMPLETED]
    if args.penalize_invalid_tool_args:
        valid_finish_reasons.append(FinishReason.INVALID_TOOL_ARGS)

    if result.metadata.finish_reason in valid_finish_reasons:
        last_message = result.traj[-1]
        assert isinstance(last_message, AIMessage), "Last message should be an AI message"
        traj_id = last_message.response_metadata["id"]
        response = requests.get(f"{sglang_endpoint}/trajectory/{traj_id}", timeout=60)
        response.raise_for_status()
        traj = response.json()
        token_ids:list[int] = traj["token_ids"]
        output_token_mask:list[int] = traj["output_token_mask"]
        token_logprobs:list[float] = traj["token_logprobs"]
        traj = traj['trajectory']
        assert len(token_ids) == len(output_token_mask), "Token ids and output token mask should have the same length"
        reward = result.metadata.metrics.get("score", 0.0)
            

        sample.tokens = token_ids
        sample.response = state.tokenizer.decode(token_ids, skip_special_tokens=False)
        sample.metadata["traj"] = traj
        sample.metadata["round_number"] = result.metadata.metrics.get("llm_calls", 0)
        #sample.response = result.traj
        try:
            first_response_idx = output_token_mask.index(1)
        except ValueError:
            first_response_idx = None
        
        if first_response_idx is None:
            logger.warning(f"No response tokens found in trajectory {traj_id}, all masks are zero")
            sample.loss_mask = []
            sample.response_length = 0
            if args.use_tis or args.use_rollout_logprobs:
                sample.rollout_log_probs = []
        else:
            sample.response_length = len(token_ids) - first_response_idx
            sample.loss_mask = output_token_mask[first_response_idx:]
            if args.use_tis or args.use_rollout_logprobs:
                sample.rollout_log_probs = token_logprobs[first_response_idx:]
        
        assert len(sample.loss_mask) == sample.response_length, \
            f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}"

        if sample.metadata.get("previous_token_len") is not None and args.mask_offpolicy_in_partial_rollout:
            previous_token_len = sample.metadata["previous_token_len"]
            assert previous_token_len <= len(token_ids), "Previous token length should be less than the current token length"
            valid_token_len = len(token_ids) - previous_token_len
            loss_mask = torch.tensor(sample.loss_mask)
            loss_mask[:-valid_token_len] = 0
            sample.loss_mask = loss_mask.tolist()
        
        if result.metadata.finish_reason == FinishReason.INVALID_TOOL_ARGS:
            response_spans = get_consecutive_span(sample.loss_mask)
            if len(response_spans) > 1:
                #Only the last turn that outputs invalid tool args will be penalized
                last_turn_span = response_spans[-1]
                loss_mask = torch.zeros(len(sample.loss_mask))
                loss_mask[last_turn_span[0]:last_turn_span[1]] = 1
                loss_mask = loss_mask.tolist()
                assert len(loss_mask) == len(sample.loss_mask), "Loss mask length should be the same"
                sample.loss_mask = loss_mask
                reward = args.invalid_tool_args_penalty

        
        sample.reward = reward
        sample.status = Sample.Status.COMPLETED
        #clean cache
        await clear_cache()
    elif result.metadata.finish_reason in [FinishReason.ABORTED]:
        last_ai_message = None
        for msg in reversed(result.traj):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        if last_ai_message is not None and args.mask_offpolicy_in_partial_rollout:
            traj_id = last_ai_message.response_metadata["id"]
            response = requests.get(f"{sglang_endpoint}/trajectory/{traj_id}", timeout=60)
            response.raise_for_status()
            traj = response.json()
            output_token_mask = traj["output_token_mask"]
            sample.metadata["previous_token_len"] = len(output_token_mask)
        if not evaluation:
            sample.status = Sample.Status.ABORTED
            sample.metadata["staleness"] = sample.metadata.get("staleness", 0) + 1
            if sample.metadata["staleness"] > args.max_staleness:
                sample.status = Sample.Status.FAILED
                sample.reward = 0.0
                return sample
            #Only when state.aborted is True, the sample will be collected into the data buffer
            #This makes sure failed samples are collected into the data buffer.
            while not state.aborted:
                await asyncio.sleep(1)
        else:
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
    else:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
    return sample

def get_consecutive_span(arr: list[int]) -> list[tuple[int, int]]:
    padded_mask = torch.tensor([0] + arr + [0])
    diff = padded_mask[1:] - padded_mask[:-1]
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    assert len(starts) == len(ends)
    return list(zip(starts, ends, strict=False))