from argparse import Namespace
import logging
from agent_core.verifiers.base import VerifyInput
from slime.utils.types import Sample
from slime.rollout.sglang_rollout import GenerateState
from typing import Any
from agent_core.runner import execute_task
from agent_core.protocol import TaskInput, EndpointConfig, ModelConfig, FinishReason
from agent_core import SQLiteSession
from agent_core.verifiers import GeneralQAVerifier
import aiohttp
import uuid


logger = logging.getLogger(__name__)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    state = GenerateState(args)
    config = args.agent_core_config
    sglang_endpoint = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    assert isinstance(sample.prompt, str), "Prompt should be a string"
    assert sample.label is not None, "Label should not be None"
    assert (
        sample.status in [Sample.Status.PENDING, Sample.Status.ABORTED, Sample.Status.FAILED]
    ), f"Sample status is {sample.status}"


    query = sample.prompt
    session_id = None
    if sample.status == Sample.Status.ABORTED:
        #resume from session
        assert 'traj_id' in sample.metadata, "Traj id is not in metadata"
        session_id = sample.metadata['session_id']
        session = SQLiteSession(session_id)
        history = await session.get_items()
        assert len(history) > 0, "Session history is empty"
        assert history[-1].get('role') != 'assistant', "Last message is from assistant"
        session.close()
        # Trajectory will resume from session history
        query = []
    
    traj_id = sample.metadata.get('traj_id', uuid.uuid4().hex)
    
    task_input = TaskInput(
        thread_id="",
        query=query,
        endpoint_cfg=EndpointConfig(
            base_url=[f"{sglang_endpoint}/v1"],
            api_key="",
            model="default",
            mode="chat-completion"
        ),
        tools=config["tools"],
        model_cfg=ModelConfig(
            system_prompt=config["system_prompt"],
            temperature=sampling_params['temperature'] if not evaluation else None,
            max_context_length=sampling_params['max_new_tokens'],
            extra_body={
                "traj_id": traj_id,
            },
        ),
        workflow=config["workflow"],
        session_id=session_id,
    )


    result = await execute_task(task_input, keep_session=True)


    async def clear_cache():
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{sglang_endpoint}/trajectory/{traj_id}") as response:
                response.raise_for_status()
        session = SQLiteSession(result.metadata.session_id)
        await session.clear_session()
        session.close()

    if result.metadata.error_info is not None:
        logger.warning(f"Rollout error: {result.metadata.error_info}")

    if result.metadata.finish_reason in [FinishReason.COMPLETED, FinishReason.INVALID_TOOL_ARGS, FinishReason.FAILED_TOOL_CALL]:
        if result.metadata.finish_reason == FinishReason.FAILED_TOOL_CALL:
            logger.error(f"Failed tool call: {result.metadata.error_info}")

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{sglang_endpoint}/trajectory/{traj_id}") as response:
                traj = await response.json()
        token_ids:list[int] = traj["token_ids"]
        output_token_mask:list[int] = traj["output_token_mask"]
        assert len(token_ids) == len(output_token_mask), "Token ids and output token mask should have the same length"
        reward = 0.0
        if result.metadata.finish_reason == FinishReason.COMPLETED:
            final_output = result.metadata.final_output
            try:
                verify_result = await GeneralQAVerifier.verify(
                    VerifyInput(
                        question=sample.prompt,
                        answer=final_output,
                        ground_truth=sample.label,
                    )
                )
                reward = verify_result.score
            except Exception as e:
                logger.error(f"Failed to verify: {e}")

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
    elif result.metadata.finish_reason == FinishReason.ABORTED:
        sample.status = Sample.Status.ABORTED
        sample.metadata['traj_id'] = traj_id
        sample.metadata['session_id'] = session_id

    return sample