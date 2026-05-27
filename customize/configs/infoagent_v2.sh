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
export FLASHINFER_WORKSPACE_BASE="/tmp/xiaolong"
export TRITON_HOME="/tmp/xiaolong/.triton"
export TRITON_CACHE_DIR="/tmp/xiaolong/.triton/cache"
rm -f /tmp/xiaolong/agent_core_session.sqlite
BASE_DIR=$(pwd)
# DATA_DIR="/data/gongrui"
# OUT_DIR="/mnt/gnet/xiaolong"
BLOB_DIR="/mnt/gnet/xiaolong"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# source "${SCRIPT_DIR}/models/tongyi_dr.sh"
# source "${SCRIPT_DIR}/models/qwen3-4b-8k.sh"
source "${SCRIPT_DIR}/models/qwen3.5-4b-32k.sh"
# source "${SCRIPT_DIR}/models/qwen3.5-4b-48k.sh"


EXP_NAME="rl_qwen3.5_4b_infoagent_v2_32k"

GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CKPT_ARGS=(
#    --hf-checkpoint $DATA_DIR/ckpts/RE-TRAC-30B-A3B
#    --hf-checkpoint $DATA_DIR/ckpts/Qwen3-4B-Instruct-2507
   --hf-checkpoint $BLOB_DIR/ckpts/Qwen3.5-4B
#    --hf-checkpoint $OUT_DIR/ckpts/Qwen3.5-4B-FP8
#    --ref-load $DATA_DIR/ckpts/RE-TRAC-30B-A3B_torch_dist
#    --ref-load $OUT_DIR/ckpts/Qwen3-4B-Instruct-2507_torch_dist
   --ref-load $BLOB_DIR/ckpts/Qwen3.5-4B_torch_dist
   --load $BLOB_DIR/slime/$EXP_NAME
   --save $BLOB_DIR/slime/$EXP_NAME
   --save-interval 10
)


ROLLOUT_ARGS=(
#    --prompt-data $DATA_DIR/data/web_47k_nosft.jsonl
#    --prompt-data $OUT_DIR/datasets/browsecomp_remaining.jsonl
#    --prompt-data $BLOB_DIR/datasets/OpenSeeker-v1-Data/openseeker_v1_data_qa.jsonl
#    --prompt-data $BLOB_DIR/datasets/REDSearcher_RL_1K.jsonl
#    --prompt-data $BLOB_DIR/datasets/tempcomp_v13_2_677.jsonl
   --prompt-data $BLOB_DIR/datasets/infoagent_v2_1000.jsonl
   --input-key question
   --label-key golden_answers
   --rollout-shuffle
   --num-rollout 300
#    --rollout-batch-size 32
#    --n-samples-per-prompt 16
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-temperature 1
#    --sglang-server-concurrency 48 # Total concurrency = server_concurrency * sglang_dp_size
   --sglang-server-concurrency 512
#    --over-sampling-batch-size 64
   --over-sampling-batch-size 96


   --num-steps-per-rollout 1
   --balance-data

   --dynamic-sampling-filter-path customize.filters.drop_invalid_samples.validate_samples
#    --dynamic-sampling-filter-path customize.filters.drop_invalid_samples.validate_samples_group_acc_lt_50
   --rollout-all-samples-process-path customize.filters.drop_invalid_samples.log_all_samples

   --custom-config-path $BASE_DIR/customize/configs/agent/tongyi_react.yaml
   --custom-generate-function-path customize.rollout.agent_core_gen.generate
   --partial-rollout
)


EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 10
   --eval-prompt-data bc300 $BLOB_DIR/datasets/browsecomp_300.jsonl
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
   --wandb-project slime-dev
   --wandb-group $EXP_NAME
   --wandb-key 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
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

   --dump-details $BLOB_DIR/slime_dump/$EXP_NAME
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
    \"PYTHONPATH\": \"$BASE_DIR/Megatron-LM:$BASE_DIR/customize\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
   \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\"
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