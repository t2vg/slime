source scripts/models/qwen3-4B-Instruct-2507.sh

PYTHONPATH=$(pwd)/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint blob/rg/ckpts/qwen3-4bins_rg_sft_10k_lora_lr2e4_bs128_ep3/v0-20260320-105228/checkpoint-237-merged \
    --save blob/rg/ckpts/qwen3-4bins_rg_sft_10k_lora_lr2e4_bs128_ep3_torch_dist