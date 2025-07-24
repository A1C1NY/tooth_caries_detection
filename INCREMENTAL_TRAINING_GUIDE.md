# 增量训练使用指南

## 问题回答：多次运行训练是否能提高效果？

**原始代码的问题：**
- 每次运行都会重新初始化模型权重
- 丢失之前的训练进展
- 无法实现真正的增量学习

**解决方案：**
我已经为您的代码添加了增量训练功能，现在可以通过多次运行来持续改进模型性能。

## 新增功能

### 1. 检查点保存和加载
- 自动保存训练状态（模型权重、优化器状态、学习率调度器状态）
- 支持从上次训练的检查点继续训练
- 保存最佳模型的独立副本

### 2. 增量训练脚本
- `incremental_training.py`：自动化多轮训练
- 智能分析训练进展
- 早停机制防止过拟合

## 使用方法

### 🚀 快速开始（推荐新手）

```bash
python quick_start_training.py
```

这个脚本会引导您选择训练模式，自动处理所有细节。

### 方法一：手动增量训练

1. **第一次训练**
```bash
python train.py --config config/config.yaml
```

2. **继续训练**
```bash
python train.py --config config/config.yaml --resume
```

3. **重复步骤2**直到满意的效果

### 方法二：自动增量训练

#### 选项1：完全自动化（后台运行）
```bash
python incremental_training.py --config config/config.yaml --rounds 5 --resume
```

#### 选项2：交互式训练（实时显示输出）
```bash
python simple_incremental_training.py --config config/config.yaml --rounds 5
```

**推荐使用选项2**，因为可以：
- 实时看到训练进度和日志
- 在每轮之间手动确认是否继续
- 随时用 Ctrl+C 中断训练
- 更好地监控训练状态

参数说明：
- `--rounds 5`：进行5轮训练
- `--min-improvement 0.001`：最小改进阈值

## 训练策略建议

### 1. 渐进式训练
```bash
# 第一阶段：快速训练
python train.py --config config/config.yaml

# 第二阶段：降低学习率继续训练
# 修改配置文件中的 base_lr 为原来的 0.1倍
python train.py --config config/config.yaml --resume

# 第三阶段：进一步微调
# 再次降低学习率
python train.py --config config/config.yaml --resume
```

### 2. 数据增强策略调整
每轮训练可以调整不同的数据增强策略：

```python
# 在 train.py 中的 get_transform 函数
def get_transform(train=True, round_num=1):
    transforms = []
    transforms.append(T.ToTensor())
    
    if train:
        if round_num == 1:
            # 第一轮：基础增强
            transforms.append(T.RandomHorizontalFlip(0.5))
        elif round_num == 2:
            # 第二轮：更强的增强
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
        # ... 更多轮次的配置
    
    return T.Compose(transforms)
```

### 3. 学习率衰减策略
```yaml
# config.yaml
training:
  # 第一轮
  base_lr: 0.00025
  
  # 第二轮（手动修改）
  # base_lr: 0.000025
  
  # 第三轮（手动修改）
  # base_lr: 0.0000025
```

## 监控训练进展

### 1. 实时监控（推荐）
使用 `simple_incremental_training.py` 可以实时看到：
```bash
python simple_incremental_training.py --config config/config.yaml --rounds 3
```

输出示例：
```
🚀 开始第 1/3 轮训练
⏰ 时间: 2025-07-24 14:30:15
============================================================
执行命令: python train.py --config config/config.yaml
============================================================

开始PyTorch牙齿龋齿检测模型训练
PyTorch版本: 2.0.1
CUDA可用: True
创建数据集...
数据集加载完成: 120 张图像
...
✅ 第 1 轮完成，基准精度: 0.7542

⏸️ 第 1 轮训练完成
按 Enter 继续下一轮训练，或输入 'q' 退出:
```

### 2. 查看训练日志
```bash
# 查看最新的训练日志
tail -f experiments/logs/training_*.log
```

### 4. 分析检查点
```python
import torch

# 加载最佳检查点
checkpoint = torch.load("experiments/detectron2_output/checkpoint_best.pth")
print(f"最佳精度: {checkpoint['metrics']['precision']:.4f}")
print(f"训练轮次: {checkpoint['epoch'] + 1}")
```

### 5. 可视化训练曲线
```python
# 可以添加到 visualize.py 中
import matplotlib.pyplot as plt

def plot_training_progress(log_files):
    """绘制多轮训练的进展曲线"""
    # 解析日志文件，提取损失和精度数据
    # 绘制训练曲线
    pass
```

## 何时停止训练

### 自动停止条件（已内置）
1. **精度改进小于阈值**：连续2轮改进 < 0.001
2. **达到最大轮数**：用户设定的最大轮数
3. **训练失败**：发生错误时自动停止

### 手动判断标准
1. **验证精度不再提升**：连续3-5轮无改进
2. **过拟合迹象**：训练精度高但验证精度下降
3. **资源限制**：时间或计算资源耗尽

## 最佳实践

### 1. 保存多个版本
```bash
# 为每轮训练创建备份
cp experiments/detectron2_output/model_final.pth models/model_round_1.pth
```

### 2. 实验记录
```bash
# 记录每轮的配置和结果
echo "Round 1: lr=0.00025, precision=0.75" >> experiment_log.txt
```

### 3. 早期验证
```bash
# 每轮训练后立即评估
python evaluate.py --model experiments/detectron2_output/model_final.pth
```

## 期望效果

通过增量训练，您可以期望：

1. **精度提升**：通常每轮可提升 1-5%
2. **更稳定的收敛**：避免训练震荡
3. **更好的泛化能力**：通过不同阶段的学习策略

## 注意事项

1. **配置一致性**：确保每轮使用相同的数据配置
2. **资源管理**：长时间训练注意GPU内存和存储空间
3. **备份重要**：定期备份最佳模型和检查点

通过这些改进，现在多次运行训练**确实可以**显著提高模型效果！
