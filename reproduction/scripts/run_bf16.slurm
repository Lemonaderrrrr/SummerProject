#!/bin/bash
# bf16 对照复现:OpenVLA 满血精度在 LIBERO-Spatial 上跑更大样本,对标官方 84.7%
# 与 run_m1.slurm 的唯一变量差异:精度(bf16 而非 4-bit)、样本量(20 trials/task)
# 用法: sbatch run_bf16.slurm
#SBATCH --job-name=openvla-bf16
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1              # bf16 ~16.8GB,A30 24GB 够;仍需 Ampere 跑 flash-attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/pub/lihonz3/logs/%x-%j.out

set -euxo pipefail

module load anaconda/2024.06
module load cuda/12.2.0
eval "$(conda shell.bash hook)"
conda activate /pub/$USER/envs/openvla

export HF_HOME=/pub/$USER/hf_cache
export TMPDIR=/pub/$USER/tmp
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch,robosuite,libero,prismatic,flash_attn; from libero.libero import benchmark; print('imports OK | cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 后台采样显存(bf16 峰值)
MEMLOG=/pub/$USER/logs/mem-$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 > "$MEMLOG" &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null" EXIT

# bf16(不加 --load_in_4bit)+ 20 trials/task = 200 rollout
cd /pub/$USER/openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 20

echo "BF16_DONE_OK"
