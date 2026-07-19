# SimplerEnv Bridge eval — environment setup

Reproduces the `f1_vla_eval` conda env for closed-loop rollout eval of a
finetuned F1-VLA checkpoint in SimplerEnv/ManiSkill2. Verified end to end: a full
3-episode rollout on `PutCarrotOnPlateInScene-v0` runs to completion, the model
loads, renders on H100/H200 GPU nodes, and drives the WidowX arm (moves the
correct object, grasps it).

## Why a separate env (not the training `f1_vla` env)

The simulator stack (sapien 2.2.2, mani_skill2_real2sim) pulls deps that fight
the training env's pins (it tries to upgrade numpy to 2.x, which breaks
torch/lerobot). `f1_vla_eval` is a clone of `f1_vla` with the sim stack layered
on top and numpy held at 1.26.4.

## Steps

```bash
# 1. Clone the working training env
conda create --name f1_vla_eval --clone f1_vla

EVAL_PY=/home/3295540/.conda/envs/f1_vla_eval/bin/python
EVAL_PIP="$EVAL_PY -m pip"

# 2. SAPIEN + sim deps, holding numpy/opencv at training-compatible pins
#    (sapien's deps otherwise upgrade numpy to 2.x -> breaks torch/lerobot)
$EVAL_PIP install "sapien==2.2.2" "numpy==1.26.4" "opencv-python==4.10.0.84" "transforms3d==0.4.2"
$EVAL_PIP install "numpy==1.26.4" "gymnasium>=0.28.1,<1.0" h5py pyyaml tqdm GitPython \
    tabulate "gdown>=4.6.0" imageio "imageio[ffmpeg]" trimesh rtree ruckig
$EVAL_PIP install matplotlib mediapy   # SimplerEnv's visualization.py needs these

# 3. Editable-install the vendored simulator + benchmark (--no-deps so they
#    don't re-resolve numpy up to 2.x)
SIMPLER=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv
(cd "$SIMPLER/ManiSkill2_real2sim" && $EVAL_PIP install -e . --no-deps)
(cd "$SIMPLER" && $EVAL_PIP install -e . --no-deps)
```

## Vulkan loader (libvulkan.so.1) — the tricky part

Two hard-won lessons:

1. **Do NOT use `conda install vulkan-loader`.** Mamba silently replaces the env's
   CPython with GraalPy (JVM-based Python lacking CPython C-ABI symbols sapien's
   compiled extensions need — you get
   `libsimsense-*.so: undefined symbol: _Py_NotImplementedStruct`), corrupting
   the whole env. Extract the loader from a Debian .deb instead (no Python
   interpreter touched).

2. **Do NOT keep the .so on scratch.** The scratch auto-purge deletes files by
   mtime, and `tar x` preserves the .deb's original 2022 mtime — so the extracted
   `libvulkan.so.1.3.204` gets purged within a day, leaving a broken symlink and
   `libvulkan.so.1: cannot open shared object file`. Put the real .so in the
   conda env's `lib/` dir (in $HOME, purge-safe); sapien finds it there
   automatically with **no LD_LIBRARY_PATH needed**.

```bash
cd /tmp
curl -sL -o libvulkan1.deb \
    "http://archive.ubuntu.com/ubuntu/pool/main/v/vulkan-loader/libvulkan1_1.3.204.1-2_amd64.deb"
ar x libvulkan1.deb && tar xf data.tar.*
cp usr/lib/x86_64-linux-gnu/libvulkan.so.1.3.204 /home/3295540/.conda/envs/f1_vla_eval/lib/
touch /home/3295540/.conda/envs/f1_vla_eval/lib/libvulkan.so.1.3.204   # fresh mtime
ln -sf libvulkan.so.1.3.204 /home/3295540/.conda/envs/f1_vla_eval/lib/libvulkan.so.1
```

The NVIDIA Vulkan ICD (the GPU driver's Vulkan impl) is already on the GPU nodes
at `/etc/vulkan/icd.d/nvidia_icd.json` — nothing to install for that.

## Runtime environment variables (set in any eval sbatch/srun job)

```bash
export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""   # headless; SAPIEN warns about GLFW/X11 then correctly renders offscreen
```

libvulkan needs no env var (it lives in the conda env lib). Bridge scene assets
(`bridge_table_1_v1.glb`, etc.) are already under `MS2_REAL2SIM_ASSET_DIR` (came
with the vendored mimic-video-project copy). NOTE: that asset dir is on scratch —
if it ever gets purged, re-obtain the SimplerEnv/ManiSkill2 real_inpainting +
stage assets.

## Smoke tests

- `test_sapien.slurm` — builds one env + renders a frame; prints `SAPIEN_RENDER_TEST_PASSED`.
- `smoke_rollout.slurm` (H200) / `smoke_rollout_h100.slurm` (H100) — full 3-episode
  closed-loop rollout of the finetuned checkpoint on PutCarrotOnPlateInScene-v0.

## Note on the SimplerEnv argparse

This vendored copy's `get_args()` was customized by mimic-video to make 8
`--vam-*` args **required** (hardcoded for their VAM policy). Our `main_inference.py`
uses its own `F1VLAInference` but still reuses this argparse, so the sbatch scripts
pass dummy `--vam-* unused/0` values plus `--ckpt-path` (read only for output-dir
naming). If you refactor, replacing `get_args()` with a slimmer F1-only parser
would remove that wart.
