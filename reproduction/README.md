# reproduction · 技术复现线(Lihong)

> 具身智能两条 VLA 路线的仿真复现 + 成本测算。
> 详细路线见 [`docs/路线图-技术线-Lihong.md`](../docs/路线图-技术线-Lihong.md)。

## 这条线要产出什么
- 在 **LIBERO** 仿真基准上跑 **OpenVLA**(token 预测)与 **π0 / openpi**(flow matching)。
- 实测并对比:成功率、推理延迟、显存、模型大小、(微调)GPU 工时。
- 把成本数据按约定格式交给报告线(见 `docs/协作工作流.md` 的交接接口契约)。

## 目录
```
reproduction/
├── README.md          # 本文件
├── 环境搭建.md         # 阶段 A:HPC3 + 环境 + 跑通第一个推理(从这开始)
├── HPC3使用教程.md     # HPC3 通用使用流程(登录 / module / Slurm)
└── scripts/
    └── hpc3_gpu_job.slurm.example   # UCI HPC3 Slurm 作业模板
```

## 现在从哪开始
👉 阶段 A:照 [`环境搭建.md`](./环境搭建.md) 一步步来,目标是**跑通一个预训练 OpenVLA 在 LIBERO 上的推理**(= 里程碑 M1)。

## 阶段(对应路线图)
- **P0 学习铺垫**:补 PyTorch/VLA、申请 HPC3、装环境、跑通推理 ← 现在
- **P1 推理评测**:OpenVLA vs π0 在 LIBERO 上对比
- **P2 LoRA 微调**:OpenVLA 微调一个任务,实测 GPU 工时
- **P3 成本建模**:汇总成本模型
