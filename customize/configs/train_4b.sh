#!/bin/bash


# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python



set -ex

ulimit -n 1048576

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export FLASHINFER_WORKSPACE_BASE="/tmp/gongrui"
export TRITON_HOME="/tmp/gongrui"
rm -f /tmp/agent_core_session.sqlite
BASE_DIR=$(pwd)

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4b-32k.sh"


EXP_NAME="qwen3-4b-rg_web_grpo"

export WANDB_JOB_NAME=$EXP_NAME
export WANDB_NAME=$EXP_NAME
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/blob/rg/ckpts/qwen3-4b-grpo179_rg_sft_ppo_w0.8_rmsparsestyleclip0.1_sg1sl1_rg1rl1_rmrb_st1_rt2_lp1.2_continue_from_w1.0step29/actor/hf/iter_0000069
   --load $BASE_DIR/blob/rg/ckpts/rl_dr/$EXP_NAME
   --save $BASE_DIR/blob/rg/ckpts/rl_dr/$EXP_NAME
   --save-interval 10
)


ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/blob/rg/data/rl_dr/web_47k_nosft.jsonl
   --input-key question
   --label-key answer
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --rollout-temperature 0.7
   --sglang-server-concurrency 96 # Total concurrency = server_concurrency * sglang_dp_size
   --over-sampling-batch-size 48


   --num-steps-per-rollout 1
   --balance-data

   --dynamic-sampling-filter-path customize.filters.drop_invalid_samples.validate_samples
   --rollout-all-samples-process-path customize.filters.drop_invalid_samples.log_all_samples

   --custom-config-path $BASE_DIR/customize/configs/agent/glm_react.yaml
   --custom-generate-function-path customize.rollout.agent_core_gen.generate
   --partial-rollout
)


EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 10
   --eval-prompt-data bc300 $BASE_DIR/blob/benchmarks/browsecomp_300.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 64000
)


ALG_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-rollout-logprobs
)


OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98


   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)


WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-4b
   --wandb-key 9aeddea3b60542704fd5cd44d4c4a1d1d911ce54
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   #--router-retry-max-retries 1
   --sglang-tool-call-parser qwen
   --log-multi-turn
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPU_NUM --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$BASE_DIR/../Megatron-LM:$BASE_DIR/customize\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPU_NUM \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${ALG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}