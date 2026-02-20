from slime.utils.types import Sample
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.logging_utils import log

def validate_samples(args, group: list[Sample]) -> DynamicFilterOutput:
    if all(s.reward == group[0].reward for s in group):
        return DynamicFilterOutput(keep=False, reason="Identical rewards")
    return DynamicFilterOutput(keep=True)

def log_all_samples(args, all_samples: list[list[Sample]], data_source):
    flattened_samples = sum(all_samples, [])
    avg_reward = sum(s.reward for s in flattened_samples) / len(flattened_samples)
    metrics = {
        "rollout/all_samples_reward": avg_reward,
    }
    log(args, metrics, step_key="rollout/step")