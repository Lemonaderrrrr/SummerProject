#!/bin/bash
# 构建 OpenVLA + LIBERO 环境(HPC3 free-gpu / V100 16GB)—— 实测可复现配方
# 用法: sbatch build_openvla.slurm
# 产物: /pub/$USER/envs/openvla(conda 环境)+ /pub/$USER/{openvla,LIBERO}(代码)
#        + ~/.libero/config.yaml(LIBERO 路径配置)
# 注: 本脚本编码了 2026-07 实跑时趟出来的所有依赖坑(见各 FIX 注释)。
#SBATCH --job-name=openvla-build
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pub/lihonz3/logs/%x-%j.out

set -euxo pipefail

module load anaconda/2024.06
module load cuda/12.2.0

# 缓存 / 临时目录 / 环境都放 /pub(HOME 有配额;TMPDIR 必须和 pip 缓存同一块盘,
# 否则 flash-attn 装 wheel 时跨盘 os.rename 报 Invalid cross-device link)
export PIP_CACHE_DIR=/pub/$USER/.cache/pip
export HF_HOME=/pub/$USER/hf_cache
export TMPDIR=/pub/$USER/tmp
mkdir -p /pub/$USER/envs "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

ENV=/pub/$USER/envs/openvla
[ -d "$ENV" ] || conda create -p "$ENV" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV"
pip install --upgrade pip

# --- PyTorch 2.2.0 (cu121) + OpenVLA ---
cd /pub/$USER
[ -d openvla ] || git clone https://github.com/openvla/openvla.git
cd openvla
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e .
pip install packaging ninja

# --- flash-attn 2.5.5 ---
# FIX: 不用源码编译(会因 TMPDIR 跨盘 rename 失败),直接装官方预编译 wheel。
#      cu122 wheel 兼容 torch 的 cu121 运行时(CUDA 12.x 小版本兼容)。
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# --- LIBERO 仿真 ---
cd /pub/$USER
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# FIX: 用 editable_mode=compat。LIBERO 顶层 libero/ 无 __init__.py(隐式命名空间包),
#      新版 setuptools 默认严格 finder 模式映射为空 → import libero 失败;compat 模式修好。
pip install -e /pub/$USER/LIBERO --config-settings editable_mode=compat
pip install -r /pub/$USER/openvla/experiments/robot/libero/libero_requirements.txt

# --- 依赖修正(踩坑后钉死的版本)---
# tf-datasets 把 tensorflow-metadata 拉到需要 protobuf5 的新版 → 和 tf2.15(protobuf4)冲突;
# libero_requirements 把 numpy 顶到 2.x / opencv 到 4.13(要 numpy2)→ 和 tf/robosuite(要 numpy<2)冲突。
# 求交集钉死:protobuf 4.25.3 喂饱 tf+wandb;numpy 1.26.4 喂饱 torch+tf+robosuite;opencv 降到 numpy1.x 版。
pip install "tensorflow-metadata==1.14.0"
pip install "protobuf==4.25.3"
pip install "numpy==1.26.4"
pip install "opencv-python==4.9.0.80"
# FIX: robosuite 1.4.1 只写 mujoco>=2.3.0(没封顶)→ pip 装最新 3.10.0,mj_fullM 签名变了
#      → robosuite 控制器崩 TypeError。降到同期 2.3.2。
pip install "mujoco==2.3.2" --no-deps
pip install "bitsandbytes==0.43.1"      # 4-bit 量化必需(OpenVLA 默认没带)
# FIX: OpenVLA 只写 accelerate>=0.25.0(没封顶)→ pip 装了最新的 1.14.0,和 transformers 4.40.1
#      内部行为对不上:加载 4-bit 时 dispatch_model 会去 .to() 量化模型 → ValueError。
#      降回同期版本 0.30.1。
pip install "accelerate==0.30.1" --no-deps

# --- 生成 LIBERO 配置(否则首次 import 交互式提问,batch 里 EOFError)---
# 喂 N = 不指定自定义数据集路径,用默认路径写 ~/.libero/config.yaml
echo "N" | python -c "from libero.libero import benchmark" || true

# --- 自检 ---
python - <<'PY'
import torch, robosuite, libero, prismatic, flash_attn, bitsandbytes
from libero.libero import benchmark
print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
print("flash_attn", flash_attn.__version__, "| all imports OK")
PY

echo "BUILD_DONE_OK"
