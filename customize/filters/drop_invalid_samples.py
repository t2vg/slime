from slime.utils.types import Sample
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.logging_utils import log

def validate_samples(args, group: list[Sample]) -> DynamicFilterOutput:
    if any(s.status == Sample.Status.FAILED for s in group):
        return DynamicFilterOutput(keep=False, reason="Failed samples")
    if all(s.reward == group[0].reward for s in group):
        return DynamicFilterOutput(keep=False, reason="Identical rewards")
    return DynamicFilterOutput(keep=True)

def validate_samples_group_acc_lt_50(args, group: list[Sample]) -> DynamicFilterOutput:
    if any(s.status == Sample.Status.FAILED for s in group):
        return DynamicFilterOutput(keep=False, reason="Failed samples")
    if all(s.reward == group[0].reward for s in group):
        return DynamicFilterOutput(keep=False, reason="Identical rewards")
    group_acc = sum(s.reward for s in group) / len(group)
    if group_acc >= 50:
        return DynamicFilterOutput(keep=False, reason="Group acc >= 50")
    return DynamicFilterOutput(keep=True)

def drop_failed_samples(args, group: list[Sample]) -> DynamicFilterOutput:
    if any(s.status == Sample.Status.FAILED for s in group):
        return DynamicFilterOutput(keep=False, reason="Failed samples")
    return DynamicFilterOutput(keep=True)

def log_all_samples(args, all_samples: list[list[Sample]], data_source):
    flattened_samples = sum(all_samples, [])
    avg_reward = sum(s.reward for s in flattened_samples) / len(flattened_samples)
    metrics = {
        "rollout/all_samples_reward": avg_reward,
    }
    log(args, metrics, step_key="rollout/step")