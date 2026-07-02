"""OpenVLA 推理延迟:计时 predict_action(dummy 224x224 输入)。延迟与输入无关(定 shape 定算量)。"""
import time, numpy as np, torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CKPT = "openvla/openvla-7b-finetuned-libero-spatial"
proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    CKPT, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
    load_in_4bit=True, low_cpu_mem_usage=True, trust_remote_code=True,
)
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
inputs = proc(prompt, img).to("cuda:0", dtype=torch.bfloat16)

for _ in range(5):  # warmup
    vla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
torch.cuda.synchronize()
N = 50; t = time.time()
for _ in range(N):
    vla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
torch.cuda.synchronize()
dt = (time.time() - t) / N
print(f"[LAT] OpenVLA-7b 4-bit: {dt*1000:.1f} ms/action, {1/dt:.2f} actions/sec (每次 forward = 1 个动作)")
