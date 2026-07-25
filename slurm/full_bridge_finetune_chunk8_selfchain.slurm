#!/bin/bash
#SBATCH --job-name=f1vla_chunk8
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/full-chunk8-%j.out

# Self-chaining training. medium_gpunew caps jobs at 6h but 100k steps at
# chunk_size=8 take ~30h, so training must run as several waves. Instead of an
# external monitor resubmitting each wave (which dies with the interactive
# session and leaves training idle for hours), this job resubmits *itself*:
# --signal=B:TERM@120 makes SLURM deliver SIGTERM to this batch script 120s
# before the wall-clock limit; the trap catches it, launches the next wave from
# the latest checkpoint, and lets this one exit. It's entirely cluster-side, so
# it keeps going with no session, no monitor, no cron. The chain stops on its
# own once checkpoint-100000 (the final step) exists.

SCRIPT=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/full_bridge_finetune_chunk8_selfchain.slurm
OUTDIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune_chunk8

resubmit_if_unfinished() {
  if [ -d "$OUTDIR/checkpoint-100000" ]; then
    echo ">>> training complete (checkpoint-100000 exists); not resubmitting."
  else
    echo ">>> time limit approaching and training unfinished -> resubmitting next wave"
    sbatch "$SCRIPT"
  fi
}
trap 'resubmit_if_unfinished' TERM

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export WANDB_PROJECT=f1-vla-bridge
export WANDB_RUN_ID=bridge_finetune_chunk8
export WANDB_RESUME=allow
# Compute nodes have no outbound internet (DNS resolution fails). With a pinned
# RUN_ID + resume, wandb must contact the server to look the run up, so
# wandb.init() blocked for 90s and then raised CommError, which killed the whole
# training process mid-wave (job 603450, exited after 1h13m having saved nothing).
# Offline mode writes metrics to disk instead; sync later from the login node
# with `wandb sync wandb/offline-run-*`. Training must never depend on an
# external service being reachable.
export WANDB_MODE=offline
cd /mnt/beegfsnew/scratch/3295540/F1-VLA

if ls outputs/bridge_finetune_chunk8/checkpoint-* >/dev/null 2>&1; then
  OVERWRITE=False
  echo ">>> checkpoint found -> resuming (overwrite_output_dir=False)"
else
  OVERWRITE=True
  echo ">>> no checkpoint -> clean start (overwrite_output_dir=True)"
fi

# Run training in the background and `wait` on it, so the SIGTERM trap can fire
# while we're blocked (a foreground child would swallow the signal).
/home/3295540/.conda/envs/f1_vla/bin/python train_hf.py \
  --config-file f1_vla/config/bridge_finetune_chunk8.yaml \
  exp.training_args.bf16=True \
  exp.training_args.report_to=[tensorboard,wandb] \
  exp.training_args.overwrite_output_dir=$OVERWRITE &
TRAIN_PID=$!
wait $TRAIN_PID

# Reached only if training exited on its own (finished 100k, or crashed) before
# the time limit — the TERM trap never fired, so nothing was resubmitted.
# Distinguish the two cases loudly: a crash silently breaking the chain is what
# happened with job 603450 (wandb CommError), and it cost hours of idle time
# before anyone noticed. Deliberately does NOT auto-resubmit: a crash that
# repeats would spawn an infinite job loop.
RC=$?
if [ -d "$OUTDIR/checkpoint-100000" ]; then
  echo ">>> TRAINING FINISHED: checkpoint-100000 reached."
else
  LAST=$(ls -d "$OUTDIR"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1)
  echo ">>> CHAIN BROKEN: training exited (rc=$RC) before the time limit at checkpoint-${LAST:-none}."
  echo ">>> Not auto-resubmitting (a repeating crash would loop). Inspect this log, fix, then resubmit:"
  echo ">>>   sbatch $SCRIPT"
fi
