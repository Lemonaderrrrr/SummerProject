#!/bin/bash
# M1: OpenVLA 4-bit 在 LIBERO-Spatial 上跑通短评测(HPC3 free-gpu / V100 16GB)
# 前提: 环境已由 build_openvla.slurm + 手动修依赖装好在 /pub/$USER/envs/openvla,
#        LIBERO 配置已生成在 ~/.libero/config.yaml
# 用法: sbatch run_m1.slurm
#SBATCH --job-name=openvla-m1
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1              # ⚠️ 必须 Ampere+:OpenVLA 用 flash-attn2,V100(Volta)跑不了
                                     #    free-gpu 上 A30(24GB)也免费、节点更多
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/pub/lihonz3/logs/%x-%j.out

set -euxo pipefail

module load anaconda/2024.06
module load cuda/12.2.0
eval "$(conda shell.bash hook)"
conda activate /pub/$USER/envs/openvla

export HF_HOME=/pub/$USER/hf_cache
export TMPDIR=/pub/$USER/tmp
export TOKENIZERS_PARALLELISM=false
# 无头节点 MuJoCo 离屏渲染走 EGL(GPU)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# --- 拿到 GPU + import 自检 ---
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch,robosuite,libero,prismatic,flash_attn,bitsandbytes; from libero.libero import benchmark; print('imports OK | cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- 后台采样显存(每 5s,拿 4-bit 峰值占用)---
MEMLOG=/pub/$USER/logs/mem-$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 > "$MEMLOG" &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null" EXIT

# --- M1 短评测(num_trials_per_task=3 快速跑通;完整 50 留 P1)---
cd /pub/$USER/openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --load_in_4bit True \
  --center_crop True \
  --num_trials_per_task 3

echo "M1_DONE_OK"
