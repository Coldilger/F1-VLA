#!/bin/bash
#SBATCH --job-name=f1_train_probes
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/probes-%j.out

set -u
DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing
PY=/home/3295540/.conda/envs/f1_vla_eval/bin/python
cd "$DIR"

for VARIANT in finetuned pretrained; do
  echo ""
  echo "############ $VARIANT ############"
  "$PY" train_probe.py \
    --features "features_${VARIANT}.npz" \
    --json-out "probe_${VARIANT}.json"
done

echo ""
echo "############ side by side ############"
"$PY" - <<'PY'
import json

rows = []
for v in ("finetuned", "pretrained"):
    with open(f"probe_{v}.json") as f:
        rows.append(json.load(f))

hdr = f"{'variant':<12}{'ridge L1':>12}{'MLP L1':>12}{'mean-pred':>12}{'no-motion':>12}{'beats mean':>12}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['variant']:<12}{r['ridge_val_l1']:>12.5f}{r['mlp_val_l1']:>12.5f}"
        f"{r['mean_pred_val_l1']:>12.5f}{r['nomotion_val_l1']:>12.5f}"
        f"{str(r['beats_mean_predictor']):>12}"
    )

ft, pre = rows[0]["ridge_val_l1"], rows[1]["ridge_val_l1"]
print()
print(f"finetuned - pretrained: {ft - pre:+.5f}  "
      f"({'finetuned better' if ft < pre else 'pretrained better'})")
PY

echo "PROBES_EXIT=$?"
