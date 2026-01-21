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
pkill -9 redis


set -ex

ulimit -n 1048576

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export FLASHINFER_WORKSPACE_BASE="/workspace/gongrui"
rm /tmp/agent_core_session.db
BASE_DIR=$(pwd)

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4b-64k.sh"


EXP_NAME="test"

GPU_NUM=8

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/ckpts/qwen4bthinking_sft_tongyi20k_lr2e5_bs512_ep5
   --ref-load $BASE_DIR/ckpts/qwen4bthinking_sft_tongyi20k_lr2e5_bs512_ep5_torch_dist
   --load $BASE_DIR/ckpts/$EXP_NAME
   --save $BASE_DIR/ckpts/$EXP_NAME
   --save-interval 20
)


ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/data/sft_qa_35k.jsonl
   --input-key question
   --label-key answer
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-temperature 1
   --sglang-server-concurrency 64 # Total concurrency = server_concurrency * sglang_dp_size
   --over-sampling-batch-size 256


   --num-steps-per-rollout 1
   --balance-data


   --custom-config-path $BASE_DIR/customize/configs/agent/tongyi_dr.yaml
   --custom-generate-function-path customize.rollout.agent_core_gen.generate
   --partial-rollout
)


EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 50
   --eval-prompt-data bc300 $BASE_DIR/data/browsecomp_300.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 64000
)


ALG_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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
   #--use-wandb
   --wandb-project slime-dev
   --wandb-group qwen4b-thinking-sft-tongyi20k-lr2e5-bs512-ep5
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
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPU_NUM --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$BASE_DIR/Megatron-LM:$BASE_DIR/customize\",
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