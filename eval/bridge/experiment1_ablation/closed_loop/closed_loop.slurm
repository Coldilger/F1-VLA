#!/bin/bash
#SBATCH --job-name=f1_exp1_cl
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/F1-VLA/eval/bridge/experiment1_ablation/closed_loop/cl-%j.out

# Experiment 1 closed loop, one script for every condition so all share
# identical settings and episodes (paired comparison via ../../paired_stats.py,
# thesis protocol 4.4). Rerun 2026-10 after the scratch purge lost the
# per-episode results. Required env vars: TASK, SEED, CONDITION
# (baseline | zero | shuffle). Save tag: exp1_${CONDITION}_seed${SEED}.

set -u
: "${TASK:?set TASK}"
: "${SEED:?set SEED}"
: "${CONDITION:?set CONDITION}"
case "$CONDITION" in
  baseline) ENTRY=/home/3295540/F1-VLA/eval/bridge/main_inference.py ;;
  zero)     ENTRY=/home/3295540/F1-VLA/eval/bridge/experiment1_ablation/main_inference_ablated.py ;;
  shuffle)  ENTRY=/home/3295540/F1-VLA/eval/bridge/experiment1_ablation/main_inference_shuffled.py ;;
  *) echo "unknown CONDITION: $CONDITION" >&2; exit 2 ;;
esac
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

/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  "$ENTRY" \
  --f1-checkpoint-path /home/3295540/F1-VLA/outputs/bridge_finetune \
  --f1-stats-path /home/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json \
  --f1-seed "$SEED" \
  --ckpt-path /home/3295540/F1-VLA/outputs/bridge_finetune \
  --robot "$ROBOT" --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps "$MAX_STEPS" \
  --env-name "$TASK" --scene-name "$SCENE" \
  --rgb-overlay-path "$OVERLAY" \
  --robot-init-x $INIT_X $INIT_X 1 --robot-init-y $INIT_Y $INIT_Y 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --additional-env-save-tags "exp1_${CONDITION}_seed${SEED}" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name unused --vam-video-model-path unused --vam-action-model-path unused \
  --vam-dataset-statistics-path unused --vam-img-horizon 0 --vam-lowdim-horizon 0 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 0

echo "EVAL_EXIT=$? TASK=$TASK SEED=$SEED CONDITION=$CONDITION"
