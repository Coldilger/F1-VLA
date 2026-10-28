#!/bin/bash
#SBATCH --job-name=repoint_editable
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3295540/F1-VLA/slurm/logs/repoint-editable-%j.out
# The repos moved from /mnt/beegfsnew/scratch/3295540 to /home/3295540 after
# the 2026-10 scratch purge, but the conda envs' editable installs
# (`pip install -e`) still pointed at the old, now-empty scratch copies
# (ModuleNotFoundError: mani_skill2_real2sim). Re-registers each editable
# package at its /home path with --no-deps, so nothing else in the envs
# changes. robocasa's editables point at scratch/robocasa-eval on purpose
# (reinstalled there by LDA-1B/eval/robocasa/install_robocasa.slurm).
set -euo pipefail
E=/home/3295540/.conda/envs
SIM=/home/3295540/F1-VLA/eval/bridge/SimplerEnv

for env in f1_vla_eval lda_eval; do
  "$E/$env/bin/python" -m pip install --no-deps -e "$SIM/ManiSkill2_real2sim"
  "$E/$env/bin/python" -m pip install --no-deps -e "$SIM"
done
"$E/f1_vla_eval/bin/python" -m pip install --no-deps -e /home/3295540/F1-VLA/f1_vla
for env in lda lda_eval; do
  "$E/$env/bin/python" -m pip install --no-deps -e /home/3295540/LDA-1B
done

echo "=== check: no editable install left pointing at the old scratch repos ==="
# only the recorded install source matters; the editable finders legitimately
# map outputs/, checkpoints/, pretrained/ to scratch (they are symlinks there)
grep -l 'scratch/3295540/\(F1-VLA\|LDA-1B\)' \
  $E/{f1_vla_eval,lda,lda_eval}/lib/python3.10/site-packages/*.dist-info/direct_url.json 2>/dev/null \
  && { echo "STALE REFS LEFT"; exit 1; } || echo "none left"
for env in f1_vla_eval lda_eval; do
  "$E/$env/bin/python" -c "import mani_skill2_real2sim, simpler_env; print('$env: simpler ok', simpler_env.__file__)"
done
(cd /home/3295540/F1-VLA && "$E/f1_vla_eval/bin/python" -c "import f1_vla.src.policies.f1_policy as m; print('f1_vla ok', m.__file__)")  # eval code puts the repo root on sys.path
for env in lda lda_eval; do
  "$E/$env/bin/python" -c "import lda; print('$env: lda ok', lda.__file__)"
done
echo "REPOINT_EXIT=0"
