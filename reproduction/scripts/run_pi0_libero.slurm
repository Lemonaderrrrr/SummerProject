#!/bin/bash
# π0.5-LIBERO 评测(openpi,客户端-服务器双进程,HPC3 free-gpu / A30)
# 前提: uv 已装;openpi 已 clone 到 /pub/$USER/openpi;服务器 venv(uv sync)+ 客户端 venv(examples/libero/.venv)已建好。
# 用法: sbatch run_pi0_libero.slurm
# 对比目标: π0.5 官方 Libero-Spatial 98.8% vs OpenVLA 84.7%;20 trials/task 对齐 OpenVLA。
#SBATCH --job-name=pi0-libero
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1              # π0 推理 >8GB,A30(24GB,Ampere)够;JAX 需 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/pub/lihonz3/logs/%x-%j.out

set -uxo pipefail

export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/pub/lihonz3/.cache/uv
export UV_PYTHON_INSTALL_DIR=/pub/lihonz3/.uv-python
export OPENPI_DATA_HOME=/pub/lihonz3/.cache/openpi   # checkpoint 缓存放 /pub
export HF_HOME=/pub/lihonz3/hf_cache
mkdir -p "$OPENPI_DATA_HOME"
cd /pub/lihonz3/openpi

# --- JAX GPU 自检(CPU-only 则快速失败,别白跑)---
uv run python -c "import jax; d=jax.devices(); print('JAX devices:', d); import sys; sys.exit(0 if any(('cuda' in str(x).lower()) or ('gpu' in str(x).lower()) for x in d) else 1)" \
  || { echo 'JAX NOT ON GPU — abort'; exit 1; }

# --- 显存采样 ---
MEMLOG=/pub/lihonz3/logs/mem-$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 10 > "$MEMLOG" &
SMI=$!

# --- 启动策略服务器(后台;首次从 GCS 下 pi05_libero checkpoint)---
SLOG=/pub/lihonz3/logs/pi0-server-$SLURM_JOB_ID.log
uv run scripts/serve_policy.py --env LIBERO > "$SLOG" 2>&1 &
SERVER=$!
trap "kill $SERVER $SMI 2>/dev/null" EXIT

# --- 等服务器就绪(端口 8000;含 checkpoint 下载+加载,最多 ~30min)---
echo "waiting for policy server on :8000 ..."
UP=0
for i in $(seq 1 180); do
  if (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then exec 3>&-; UP=1; echo "server up after ~$((i*10))s"; break; fi
  kill -0 $SERVER 2>/dev/null || { echo "SERVER DIED early:"; tail -40 "$SLOG"; exit 1; }
  sleep 10
done
[ "$UP" = 1 ] || { echo "server not up in time:"; tail -40 "$SLOG"; exit 1; }

# --- 客户端(独立 py3.8 venv):LIBERO-Spatial, 20 trials/task ---
source examples/libero/.venv/bin/activate
export PYTHONPATH=${PYTHONPATH:-}:$PWD/third_party/libero   # ${:-} 防 set -u 下 PYTHONPATH 未定义报错
export MUJOCO_GL=egl                  # 无头离屏渲染;若 egl 报错换 glx
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20

echo "PI0_LIBERO_DONE"
