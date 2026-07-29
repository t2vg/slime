MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --swiglu
   --num-layers 32
   --hidden-size 2560
   --ffn-hidden-size 9216
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 4
   --disable-bias-linear
   --kv-channels 256
   --qk-layernorm
   --use-gated-attention

   --normalization "RMSNorm"
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --rotary-base 10000000
   --vocab-size 248320

   # qwen3.5 specific
   --attention-output-gate


)

PYTHONPATH="$(pwd):$(pwd)/Megatron-LM" python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint blob/rg/ckpts/sft_base/qwen3.5-4b_sft_with_13k_cycle1/v0-20260602-150833/checkpoint-1300 \
    --save blob/rg/ckpts/qwen3.5-4b_sft_with_13k_cycle1_torch_dist