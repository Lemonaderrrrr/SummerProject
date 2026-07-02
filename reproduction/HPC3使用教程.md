# UCI HPC3 使用教程

> 从登录到跑任务的完整流程。命令依据 RCIC 官方文档(2026-06 查证),确切参数以官方为准。
> 官方文档:https://rcic.uci.edu/hpc3/hpc3.html

---

## 0. 前置条件
- **UCInetID + 密码 + DUO**(手机 App 二次验证)。
- **在校园网内,或连 UCI VPN**(校外必须先连 VPN)。

## 1. 登录(SSH)
```bash
ssh <UCInetID>@hpc3.rcic.uci.edu
```
输 UCInetID 密码 → 出现 DUO 提示,输 `1` 推送到手机点批准。
> 嫌每次输密码烦,可配 SSH key(见 RCIC「Generate SSH keys」)。

## 2. 环境与规矩(重要)
- 登录后你在**登录节点(login node)**:只用来**编辑文件、提交作业**。
  ❌ **不要在登录节点跑训练/重计算**(会被管理员杀)。
- 真正算力在**计算节点**,一律通过 **Slurm** 调度。
- 存储:
  - `$HOME`:代码 / 小文件(有配额)。
  - **`/pub/<UCInetID>`:大数据集 + 模型 checkpoint**(LIBERO ~10GB、VLA 权重很大,别塞 $HOME)。

## 3. 软件:module 系统
```bash
module avail            # 看有哪些软件
module load anaconda    # 加载(名字以 module avail 为准)
```
> ⚠️ `module load` 要在**作业脚本里**或**进了交互节点之后**做,别在登录节点上 load。

## 4. 跑任务:两种方式

### A. 交互式 —— 调试 / 装环境
```bash
srun -p free-gpu --gres=gpu:1 --pty /bin/bash -i
# 进去后:module load、conda activate、手动跑命令、看报错
```
- 免费 CPU 队列:`-p free`
- 用实验室账号:`-A <lab>_lab`

### B. 批处理 —— 正式跑长任务(推荐)
用 `scripts/hpc3_gpu_job.slurm.example`:
```bash
mkdir -p logs
sbatch myjob.slurm       # 提交
squeue -u $USER          # 状态:PD=排队, R=运行中
scancel <jobid>          # 取消
cat logs/<job>.out       # 看输出
```

## 5. 本项目的典型循环
1. `ssh` 登录
2. `srun` 进一个带 GPU 的交互节点
3. `conda activate openvla`,装依赖 / 调通推理命令
4. 调通后写进 sbatch 脚本,`sbatch` 正式跑
5. `squeue` 监控,`logs/` 看结果
6. 数据 / checkpoint 存 `/pub`

## 6. 额度与成本
- 免费一次性 **1000 core-hours** + `free-gpu` 抢占队列(免费,可能被中断)。
- GPU 计费:单 GPU 最低 **34 SU/小时** → 1000 core-hours ≈ **~29 个计费 GPU 小时**。
- 查余额命令以 RCIC「Resource Allocations」页为准;不够用就找 PI 挂靠或发邮件问 recharge。

## 参考(官方)
- [Logging in(登录 + DUO + VPN)](https://rcic.uci.edu/account/login.html)
- [Beginner guide](https://rcic.uci.edu/guides/beginner.html)
- [Jobs howto(Slurm)](https://rcic.uci.edu/slurm/jobs.html) · [Job examples](https://rcic.uci.edu/slurm/examples.html)
- [Resource Allocations(额度/成本)](https://rcic.uci.edu/about/allocations.html)
