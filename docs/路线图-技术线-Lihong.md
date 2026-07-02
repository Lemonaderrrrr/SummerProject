# 路线图 · 技术线(Lihong)

> 我的产物:**技术复现 demo + 成本测算**。
> 一句话:在仿真里复现 π0 与 OpenVLA 两条 VLA 路线,实测它们的资源成本,把成本数据交给 Nicky 写报告。
> 共享时间线见 [路线图(通用)](./路线图.md);Nicky 的线见 [报告线](./路线图-报告线-Nicky.md)。

> ⚠️ 现状:我是 ML **入门**(概念懂、没真正训练过模型)。所以这条线**前期重学习、推理评测先行、微调放后面**,风险可控:就算微调没做成,推理评测阶段也已能产出对比 + 成本数据。

---

## 我会学到的新技术
PyTorch · VLA / 模仿学习(imitation learning)· 机器人仿真(LeRobot + LIBERO)· LoRA 微调 · HPC/Slurm 集群使用 · 自底向上成本建模。

## 技术选型(待 Issue #2 最终确认)
- **模型**:OpenVLA(Apache 2.0,消费级 GPU 可微调)+ openpi 的 π0 / π0.5(开源权重 + 微调代码)。
- **仿真 / 基准**:LIBERO(OpenVLA、openpi 都现成支持),框架用 LeRobot。
- **算力**:UCI HPC3,**走完全免费档**。
  - 申请:用 UCInetID 邮箱发 hpc-support@uci.edu;得免费 1000 core-hours + `free-gpu` 分区。
  - **主力**:`free-gpu`(**V100 16GB**)→ OpenVLA **4-bit 推理(~7GB)+ QLoRA 微调**(16GB 够)。
  - **π0**:推理要 ≥24GB,V100 跑不动 → 用一次性 **1000 core-hours(≈29 个 A30/A100 GPU 小时)** 在计费 `gpu` 分区省着跑几次做对比。
  - 详细搭建见 `reproduction/环境搭建.md`、`reproduction/HPC3使用教程.md`。

---

## 分阶段(P0 → P3)

### P0 · 学习铺垫(W1–2)
- [ ] 补 Python / PyTorch;读一篇 VLA 综述,搞懂 flow matching vs token 预测的区别。
- [ ] 申请 UCI HPC3 账号,跑通 Slurm 提交一个 GPU 任务。
- [ ] 装好 LeRobot + LIBERO 仿真环境。
- [ ] **跑通一个预训练 OpenVLA 在 LIBERO 上的推理**(= 里程碑 M1 的我这半)。

### P1 · 推理 + 评测(W3–4)——对比主战场
- [ ] 在同一 LIBERO 基准上跑 **OpenVLA**(免费 V100,4-bit)与 **π0(openpi)**(1000 core-hr 额度,A30)。
- [ ] 记录:任务成功率、推理延迟、显存占用、模型参数量。
- [ ] **交付 M2**:把「推理成本 + 模型规模对比」整理成表交给 Nicky。
- 👉 这一阶段不需要会训练,适合入门起步,但已能产出真实对比数据。

### P2 · LoRA 微调一个任务(W5–7)——进阶/加分
- [ ] 用 OpenVLA 在 LIBERO 的一个任务上做 LoRA 微调(单 A100,几小时)。
- [ ] 记录 GPU 工时、显存、训练曲线。
- [ ] (stretch)尝试 openpi 的 π0 微调,做真正的两路线对比。
- [ ] **交付 M3**:LoRA 微调实测 GPU 工时 → 折算 $ 训练成本,交给 Nicky。

### P3 · 成本建模(W8–9)
- [ ] 汇总:实测算力成本 + 公开训练配方(外推全规模)+ 数据成本($118/hr × 示范数)。
- [ ] 产出 **π0 vs OpenVLA 的成本模型**(训练 / 推理 / 数据三块)。

### 整合(W10)
- [ ] demo 整理(能跑起来 + README 说明)、技术部分文档化,配合 Nicky 整合最终形态。

---

## 我交给 Nicky 的东西(耦合接口)
| 里程碑 | 交付内容 |
|--------|----------|
| M2 | 推理成本、模型规模、显存/延迟对比表 |
| M3 | LoRA 微调实测 GPU 工时 → $ 训练成本 |
| P3 | 完整成本模型(训练/推理/数据)|

## 可选后续产出
- M4 双初稿完成后,可把 π0 vs OpenVLA 的成本分析整理成 **4–6 页 workshop paper**,投 **CoRL / ICRA 2027 workshop**(cost analysis 属 niche case study,适合 workshop)。不影响主线节奏,作为加分项。

## 诚实的风险点
- 光跑通两个 VLA 仓库 + 接仿真,对入门者有学习曲线 → 所以 P0/P1 多留时间。
- π0(openpi)环境比 OpenVLA 复杂 → 先把 OpenVLA 跑顺,π0 作为对比再加。
- 成本数是**带假设的估算**,不是公司内部真实数字 → 文档里标清假设。
