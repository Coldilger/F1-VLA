#!/bin/bash
# Submits the Experiment 9 (F1 scales) grid while keeping this user's queue
# under the cluster limit (30 queued jobs). Runs on the login node; it only
# calls squeue / sbatch and sleeps. Appends "<condition> <task> <seed> <jobid>"
# to jobs.txt.
cd /home/3295540/F1-VLA/eval/bridge/experiment9_scales
TASKS="PutCarrotOnPlateInScene-v0 PutSpoonOnTableClothInScene-v0 StackGreenCubeOnYellowCubeBakedTexInScene-v0 PutEggplantInBasketScene-v0"
LIMIT=28
grid=()
for SEED in 0 1 2; do
  for C in "1 1" "6 1" "2 1" "1 0" "6 0" "2 0"; do
    for T in $TASKS; do grid+=("$C $T $SEED"); done
  done
  if [ "$SEED" = 0 ]; then for T in $TASKS; do grid+=("4 0 $T 0"); done; fi
done
for item in "${grid[@]}"; do
  set -- $item
  S=$1; P=$2; T=$3; SEED=$4
  if grep -q "^scales${S}_pos${P} $T $SEED " jobs.txt 2>/dev/null; then continue; fi
  while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$LIMIT" ]; do sleep 120; done
  until J=$(TASK=$T SEED=$SEED SCALES=$S POS_MATCH=$P sbatch --parsable --export=ALL scales_closed_loop.slurm); do
    echo "sbatch failed for $item, retrying"; sleep 300
  done
  echo "scales${S}_pos${P} $T $SEED $J" >> jobs.txt
done
echo "FEEDER_DONE $(wc -l < jobs.txt) jobs"
