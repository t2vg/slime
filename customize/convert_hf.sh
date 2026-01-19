source scripts/models/qwen3-4B-Instruct-2507.sh

PYTHONPATH=$(pwd)/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ckpts/qwen4bthinking_sft_tongyi20k_lr2e5_bs512_ep5 \
    --save ckpts/qwen4bthinking_sft_tongyi20k_lr2e5_bs512_ep5_torch_dist