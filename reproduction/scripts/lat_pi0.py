"""π0.5 推理延迟:计时 policy.infer(dummy LIBERO 观测)。返回一个动作 chunk。"""
import time, numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download

cfg = _config.get_config("pi05_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
policy = _policy_config.create_trained_policy(cfg, ckpt)

element = {
    "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),   # LIBERO state 维度(错则按报错调)
    "prompt": "pick up the black bowl and place it on the plate",
}
for _ in range(3):  # warmup(含 JAX 编译)
    out = policy.infer(element)
N = 50; t = time.time()
for _ in range(N):
    out = policy.infer(element)
dt = (time.time() - t) / N
K = int(np.asarray(out["actions"]).shape[0])
print(f"[LAT] pi0.5: {dt*1000:.1f} ms/infer, chunk={K} 动作/次, 有效 {K/dt:.2f} actions/sec (replan 每 5 步)")
