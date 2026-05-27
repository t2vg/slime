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


   # rollout
   --rollout-max-response-len 64000


   # perf
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 8
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1


   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1


   #--micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 64000



   #sglang
   --rollout-num-gpus-per-engine 1
   # AssertionError: Page size must be 1 for MambaRadixCache v1, got 64
   --sglang-page-size 1
   # HiRadixCache does not support GQA yet, disabled for Qwen3.5
   #--sglang-enable-hierarchical-cache
   #--sglang-hicache-size 250
   #--sglang-hicache-io-backend kernel
   #--sglang-hicache-write-policy write_through
)
