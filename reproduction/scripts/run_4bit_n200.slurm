#!/bin/bash
# 4-bit @ 200 rollout:与 run_bf16.slurm 严格同配置(A30/libero_spatial/20 trials/同协议),
# 唯一变量 = 精度(4-bit)。目的:干净对比 4-bit vs bf16 的量化掉分。
# 用法: sbatch run_4bit_n200.slurm
#SBATCH --job-name=openvla-4bit-n200
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
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
MEMLOG=/pub/$USER/logs/mem-$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 > "$MEMLOG" &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null" EXIT

cd /pub/$USER/openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --load_in_4bit True \
  --center_crop True \
  --num_trials_per_task 20

echo "4BIT_N200_DONE_OK"
