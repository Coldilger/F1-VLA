#!/bin/bash
#SBATCH --job-name=f1_exp67_swap
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/F1-VLA/eval/bridge/experiment6_7_swap/swap-%j.out

# Experiments 6-7 (graded / horizon-shifted foresight replacement), closed
# loop: the robot executes the actions decoded with the replaced forecast
# (f1_swap_inference.py). Required env vars: TASK, SEED, SWAP
# (bank:same_task | bank:other_task | bank:other_scene | own:<d> | real:<h> | control).
# Optional: EP_START (0), EP_END (24), TAG_SUFFIX, F1_CKPT.
# Save tag: exp67_<swap>_seed<S><TAG_SUFFIX>, ':' -> '_', '+' -> 'p', '-' -> 'm'.

set -u
: "${TASK:?set TASK}"
: "${SEED:?set SEED}"
: "${SWAP:?set SWAP}"
EP_START="${EP_START:-0}"
EP_END="${EP_END:-24}"
COND=$(echo "$SWAP" | sed 's/:/_/; s/+/p/; s/-/m/')
TAG="exp67_${COND}_seed${SEED}${TAG_SUFFIX:-}"
F1_CKPT="${F1_CKPT:-/home/3295540/F1-VLA/outputs/bridge_finetune}"
export SWAP TASK
export SWAP_OUT=/home/3295540/F1-VLA/eval/bridge/experiment6_7_swap/records/${TAG}/${TASK}
ENTRY=/home/3295540/F1-VLA/eval/bridge/experiment6_7_swap/main_inference_swap.py
# RLIMIT_CPU (~600 CPU-seconds) kills sim evals mid-run without this
ulimit -t unlimited

export MS2_REAL2SIM_ASSET_DIR=/home/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

SIMPLER=/home/3295540/F1-VLA/eval/bridge/SimplerEnv
cd "$SIMPLER"

case "$TASK" in
  PutCarrotOnPlateInScene-v0|PutSpoonOnTableClothInScene-v0|StackGreenCubeOnYellowCubeBakedTexInScene-v0)
    ROBOT=widowx
    SCENE=bridge_table_1_v1
    OVERLAY=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    INIT_X=0.147; INIT_Y=0.028; MAX_STEPS=60
    ;;
  PutEggplantInBasketScene-v0)
    ROBOT=widowx_sink_camera_setup
    SCENE=bridge_table_1_v2
    OVERLAY=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
    INIT_X=0.127; INIT_Y=0.06; MAX_STEPS=120
    ;;
  *)
    echo "unknown TASK: $TASK" >&2; exit 2
    ;;
esac

echo "SWAP=$SWAP TAG=$TAG F1_CKPT=$F1_CKPT"
/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  "$ENTRY" \
  --f1-checkpoint-path "$F1_CKPT" \
  --f1-stats-path "$F1_CKPT/bridge_orig_stats.json" \
  --f1-seed "$SEED" \
  --ckpt-path "$F1_CKPT" \
  --robot "$ROBOT" --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps "$MAX_STEPS" \
  --env-name "$TASK" --scene-name "$SCENE" \
  --rgb-overlay-path "$OVERLAY" \
  --robot-init-x $INIT_X $INIT_X 1 --robot-init-y $INIT_Y $INIT_Y 1 \
  --obj-variation-mode episode --obj-episode-range "$EP_START" "$EP_END" \
  --additional-env-save-tags "$TAG" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name unused --vam-video-model-path unused --vam-action-model-path unused \
  --vam-dataset-statistics-path unused --vam-img-horizon 0 --vam-lowdim-horizon 0 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 0

echo "EVAL_EXIT=$? TASK=$TASK SEED=$SEED SWAP=$SWAP"
