#!/bin/bash


# for rerun the task
export NUM_NODES=1

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


if [ "$EXP_NAME" == "" ]; then
   echo "EXP_NAME is not set"
   exit 1
fi
RM_URL=${RM_URL:-"http://172.179.10.5:30000"}
CRITIC_NUM_HEADS=${CRITIC_NUM_HEADS:-2}
REASONABLE_REWARD_WEIGHT=${REASONABLE_REWARD_WEIGHT:-0.8}
REASONABLE_TEMPERATURE=${REASONABLE_TEMPERATURE:-2}
STYLE_TEMPERATURE=${STYLE_TEMPERATURE:-1}
STYLE_REWARD_CLIP=${STYLE_REWARD_CLIP:-0.1}
LP_MEAN=${LP_MEAN:-500}
RESUME_CKPT=${RESUME_CKPT:-0}
export WANDB_JOB_NAME=$EXP_NAME
export WANDB_NAME=$EXP_NAME
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || true)
if [ "$GPU_NUM" -eq 0 ]; then
    GPU_NUM=$(amd-smi list | grep GPU | wc -l)
fi

bash customize/launch_data_encoder.sh &

# check if the checkpoint exists
ACTOR_CKPT=""
CRITIC_CKPT=""

if [ "$RESUME_CKPT" -eq 1 ]; then
   if [ -f $BASE_DIR/blob/rg/ckpts/$EXP_NAME/actor/latest_checkpointed_iteration.txt ]; then
      ACTOR_CKPT="--load $BASE_DIR/blob/rg/ckpts/$EXP_NAME/actor"
   else
      echo "Actor checkpoint not found"
      exit 1
   fi

   if [ -f $BASE_DIR/blob/rg/ckpts/$EXP_NAME/critic/latest_checkpointed_iteration.txt ]; then
      CRITIC_CKPT="--critic-load $BASE_DIR/blob/rg/ckpts/$EXP_NAME/critic"
   else
      echo "Critic checkpoint not found"
      exit 1
   fi
fi

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/blob/rg/ckpts/qwen3-4b-grpo179_rg_sft_notag_10k_lora_lr2e4_bs128_ep3/v3-20260326-161646/checkpoint-237-merged
   --ref-load $BASE_DIR/blob/rg/ckpts/rl_dr/qwen3-4b-grpo179_rg_stage1_torch_dist
   --save $BASE_DIR/blob/rg/ckpts/$EXP_NAME/actor
   --critic-save $BASE_DIR/blob/rg/ckpts/$EXP_NAME/critic
   --save-interval 10
   $ACTOR_CKPT
   $CRITIC_CKPT
)


ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/blob/rg/data/rl_rg/22k_cycle1_havefinal_long_train_for_rg_rl.jsonl
   --input-key question
   --rollout-shuffle
   --num-rollout 80
   --rollout-batch-size 256
   --n-samples-per-prompt 2
   --rollout-temperature 1
   --sglang-server-concurrency 96 # Total concurrency = server_concurrency * sglang_dp_size
   --over-sampling-batch-size 384


   --num-steps-per-rollout 1
   --balance-data

   --dynamic-sampling-filter-path customize.filters.drop_invalid_samples.drop_failed_samples
   #--rollout-all-samples-process-path customize.filters.drop_invalid_samples.log_all_samples

   --custom-config-path $BASE_DIR/customize/configs/agent/rg.yaml
   --custom-generate-function-path customize.rollout.rg2.generate
   --partial-rollout
   --rm-url $RM_URL
)


EVAL_ARGS=(
   --skip-eval-before-train
   #--eval-interval 10
   #--eval-prompt-data bc300 $BASE_DIR/data/browsecomp_300.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 64000
)


ALG_ARGS=(
   --advantage-estimator ppo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-rollout-logprobs
   --use-customize-rewards
   --normalize-advantages
   --num-critic-only-steps 10
   --critic-num-heads $CRITIC_NUM_HEADS
   --reasonable-reward-weight $REASONABLE_REWARD_WEIGHT
   --reasonable-temperature $REASONABLE_TEMPERATURE
   --style-temperature $STYLE_TEMPERATURE
   --style-reward-clip $STYLE_REWARD_CLIP
   --lp-mean $LP_MEAN
   --gamma 1.0
   --lambd 1.0
   --gamma-reasonable 1.0
   --lambd-reasonable 1.0
   --gamma-style 1.0
   --lambd-style 1.0
)


OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --critic-lr 9e-6
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
   --wandb-project rg-rl
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
   #--sglang-tool-call-parser qwen
   --log-multi-turn
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
if [ "$NUM_NODES" -eq 1 ]; then
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPU_NUM --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi



# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$BASE_DIR/../Megatron-LM:$BASE_DIR/customize\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TRITON_HOME\": \"${TRITON_HOME}\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\",
    \"HSA_NO_SCRATCH_RECLAIM\": \"1\",
    \"AITER_JIT_DIR\": \"/tmp/aiter\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes $NUM_NODES \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${ALG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}