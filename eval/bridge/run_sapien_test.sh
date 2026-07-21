#!/bin/bash
export LD_LIBRARY_PATH=/mnt/beegfsnew/scratch/3295540/vulkan_libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
cd /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim
/home/3295540/.conda/envs/f1_vla_eval/bin/python /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/test_sapien_render.py
