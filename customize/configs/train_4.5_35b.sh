#!/bin/bash


# for rerun the task

export NUM_NODES=1
cd ~/projects/slime.worktrees/gongrui-fix
if [ "$NUM_NODES" -eq 1 ]; then
   pkill -9 sglang
   sleep 3
   ray stop --force
   pkill -9 ray
   pkill -9 python
   sleep 3
   pkill -9 ray
   pkill -9 python
fi



set -ex

ulimit -n 1048576

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export FLASHINFER_WORKSPACE_BASE="/tmp/gongrui"
export TRITON_HOME="/tmp/gongrui"
rm -f /tmp/gongrui/agent_core_session.sqlite
BASE_DIR=$(pwd)

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B-32k.sh"


if [ "$EXP_NAME" == "" ]; then
   echo "EXP_NAME is not set"
   exit 1
fi

export WANDB_JOB_NAME=$EXP_NAME
export WANDB_NAME=$EXP_NAME
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/blob/rg/ckpts/Qwen3.5-35B-A3B
   --ref-load $BASE_DIR/blob/rg/ckpts/Qwen3.5-35B-A3B_torch_dist
   #--load $BASE_DIR/blob/rg/ckpts/rl_dr/$EXP_NAME
   --save $BASE_DIR/blob/rg/ckpts/rl_dr/$EXP_NAME
   --save-interval 10
)


ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/blob/rg/data/rl_dr/bc1k_web1k.jsonl
   --input-key question
   --label-key answer
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-temperature 1
   --sglang-server-concurrency 256
   --over-sampling-batch-size 96

   --use-dynamic-global-batch-size
   --num-steps-per-rollout 1
   --balance-data

   --dynamic-sampling-filter-path customize.filters.drop_invalid_samples.validate_samples
   --rollout-all-samples-process-path customize.filters.drop_invalid_samples.log_all_samples
   #--rollout-sample-filter-path customize.rollout.retrac.flatten_round_samples
   #--custom-reward-post-process-path customize.rollout.retrac.post_process_rewards

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
   --normalize-advantages
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
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
   --wandb-project rl_dr
   --wandb-group $EXP_NAME
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
   --sglang-tool-call-parser qwen3_coder
#    --sglang-moe-runner-backend triton
#    --sglang-log-requests

   --dump-details $BASE_DIR/blob/retrac/ckpts/$EXP_NAME/dump
   --log-multi-turn
)

# launch the master node of ray in container
# Pin distributed bootstrap to the pod/node interface; NCCL may otherwise choose tun0.
export SLIME_SOCKET_IFNAME=${SLIME_SOCKET_IFNAME:-"eth0"}
if [ -z "${MASTER_ADDR:-}" ]; then
   MASTER_ADDR=$(ip -4 -o addr show dev "${SLIME_SOCKET_IFNAME}" | awk '{split($4, a, "/"); print a[1]; exit}')
   if [ -z "${MASTER_ADDR}" ]; then
      MASTER_ADDR=$(hostname -I | awk '{print $1}')
   fi
fi
export MASTER_ADDR
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"${SLIME_SOCKET_IFNAME}"}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"${SLIME_SOCKET_IFNAME}"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPU_NUM --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/workspace/gongrui/projects/Megatron-LM:$BASE_DIR/customize\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
   \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"TRITON_HOME\": \"${TRITON_HOME}\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\"
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