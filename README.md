# SummerProject

> 复现驱动的具身智能「成本 × 投资」分析 —— 一个机器人 × 金融交叉的暑假项目。

> 📌 方向已拟定(v1),最终细节待 Nicky 在 [路线图 PR](https://github.com/Lemonaderrrrr/SummerProject/pulls) 上确认。

## 这是什么

在仿真中复现具身智能(embodied AI)两条代表性技术路线 —— **Physical Intelligence π0 的 flow matching** vs **OpenVLA 的 token 预测** —— 实测它们的训练 / 推理 / 数据成本,据此产出一份「**护城河在数据、不在算力**」的行业投资分析报告。

项目有两个**互相咬合**的产物:
- **技术复现 demo + 成本测算**(Lihong,具身智能方向):复现技术栈,量出真实成本。
- **行业研究报告**(Nicky,金融方向):用这些自底向上的成本数据,做公司竞争力与投资分析。

> 耦合点:复现量出的成本 → 成为报告的定量骨架。这是这个项目的差异化 —— 报告里的成本不是二手数字,是动手算出来的。

## 目标

- **学新技术**(两人各自对口):Lihong 学 VLA / 仿真 / 模型微调;Nicky 学行业研究 / 成本建模 / 投资分析。
- 整个暑假做出**两人简历都能写**的产物。
- 范围:纯仿真、2–3 家公司、开源权重、自底向上成本估算。**不做**真机硬件、不 1:1 复刻闭源模型、不做生产系统。

## 技术栈

> 详细选型见 Issue #2;以下为拟定方向。

- **复现**:Python / PyTorch、OpenVLA + openpi(π0/π0.5)、LeRobot、仿真基准 LIBERO
- **算力**:UCI HPC3(A100 / A30)
- **报告**:行业数据抓取 + 成本建模(工具待定)

## 如何运行

```bash
# 待补充(尚无代码,见路线图阶段 A)
```

## 路线图

- [通用时间线](./docs/路线图.md) · [技术线 · Lihong](./docs/路线图-技术线-Lihong.md) · [报告线 · Nicky](./docs/路线图-报告线-Nicky.md)

## 协作方式

我们用 **GitHub Issues** 当任务看板,每人各自用自己的 Claude Code 认领任务、开分支、提 PR。
详细的协作规范见 [`CLAUDE.md`](./CLAUDE.md)。

## 参与者

- @Lemonaderrrrr(Lihong,具身智能方向 —— 技术复现)
- @NickyNg413(Nicky,金融方向 —— 行业研究报告)
