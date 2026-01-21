MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 5000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm


   # rollout
   --rollout-max-response-len 64000


   # perf
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1


   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1


   #--micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32000



   #sglang
   --rollout-num-gpus-per-engine 1
   --sglang-page-size 64
   --sglang-enable-hierarchical-cache
   --sglang-hicache-size 250
   --sglang-hicache-io-backend kernel
   --sglang-hicache-write-policy write_through
)