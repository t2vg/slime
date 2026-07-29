source scripts/models/qwen3-4B-Instruct-2507.sh

PYTHONPATH=$(pwd)/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint blob/rg/ckpts/sft_base/qwen3-4b-grpo179_sft_with_15k_gpt5.5_nothink/v0-20260601-101955/checkpoint-910 \
    --save blob/rg/ckpts/Qwen3-4B-Thinking-2507