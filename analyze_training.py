#!/usr/bin/env python3
"""
训练状态分析工具
分析当前训练进展和模型性能
"""

import os
import torch
import yaml
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re

def load_checkpoint_info(checkpoint_path):
    """加载检查点信息"""
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None
    return None

def parse_log_file(log_file):
    """解析日志文件，提取训练指标"""
    if not os.path.exists(log_file):
        return None
    
    epochs = []
    precisions = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配精度信息
            if "精度:" in line:
                try:
                    precision_match = re.search(r'精度: ([\d.]+)', line)
                    if precision_match:
                        precision = float(precision_match.group(1))
                        precisions.append(precision)
                except:
                    continue
    
    return precisions

def analyze_training_status():
    """分析训练状态"""
    print("   训练状态分析")
    print("="*60)
    
    # 1. 检查配置
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        output_dir = config['training']['output_dir']
        print(f"   输出目录: {output_dir}")
    else:
        print("   配置文件不存在")
        return
    
    # 2. 检查检查点
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    best_checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")
    
    current_checkpoint = load_checkpoint_info(checkpoint_path)
    best_checkpoint = load_checkpoint_info(best_checkpoint_path)
    
    if current_checkpoint:
        current_epoch = current_checkpoint.get('epoch', 0)
        current_metrics = current_checkpoint.get('metrics', {})
        current_precision = current_metrics.get('precision', 0)
        
        print(f"\n   当前检查点状态:")
        print(f"  Epoch: {current_epoch + 1}")
        print(f"  精度: {current_precision:.4f}")
        print(f"  总预测数: {current_metrics.get('total_predictions', 0)}")
        print(f"  正确预测数: {current_metrics.get('correct_predictions', 0)}")
    
    if best_checkpoint:
        best_epoch = best_checkpoint.get('epoch', 0)
        best_metrics = best_checkpoint.get('metrics', {})
        best_precision = best_metrics.get('precision', 0)
        
        print(f"\n   最佳检查点状态:")
        print(f"  Epoch: {best_epoch + 1}")
        print(f"  精度: {best_precision:.4f}")
        print(f"  总预测数: {best_metrics.get('total_predictions', 0)}")
        print(f"  正确预测数: {best_metrics.get('correct_predictions', 0)}")
        
        # 分析性能变化
        if current_checkpoint and best_checkpoint:
            precision_diff = current_precision - best_precision
            print(f"\n   性能变化分析:")
            if precision_diff > 0:
                print(f"     精度提升: +{precision_diff:.4f}")
            elif precision_diff < -0.01:
                print(f"      精度明显下降: {precision_diff:.4f}")
                print(f"     建议立即停止训练，可能出现过拟合！")
            else:
                print(f"      精度变化较小: {precision_diff:.4f}")
    
    # 3. 检查最新日志
    log_dir = config.get('training', {}).get('log_dir', 'experiments/logs')
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.startswith('training_') and f.endswith('.log')]
        if log_files:
            latest_log = os.path.join(log_dir, sorted(log_files)[-1])
            print(f"\n📋 最新日志文件: {latest_log}")
            
            # 解析精度变化
            precisions = parse_log_file(latest_log)
            if precisions and len(precisions) >= 2:
                recent_precisions = precisions[-5:]  # 最近5次评估
                print(f"  最近精度变化: {[f'{p:.4f}' for p in recent_precisions]}")
                
                if len(recent_precisions) >= 2:
                    trend = recent_precisions[-1] - recent_precisions[-2]
                    if trend < -0.005:
                        print(f"     精度下降趋势: {trend:.4f}")
                    elif trend > 0.005:
                        print(f"     精度上升趋势: +{trend:.4f}")
                    else:
                        print(f"     精度变化平缓: {trend:.4f}")

def provide_recommendations():
    """提供操作建议"""
    print(f"\n   操作建议:")
    print("="*60)
    
    print("1.    立即停止训练 (Ctrl+C)")
    print("   - 当前显示过拟合迹象")
    print("   - 继续训练可能导致性能进一步下降")
    
    print("\n2.    使用最佳模型进行评估:")
    print("   python train.py --config config/config.yaml --eval-only")
    
    print("\n3.    调整训练策略:")
    print("   - 降低学习率 (例如: base_lr: 0.00025 -> 0.000025)")
    print("   - 增加正则化")
    print("   - 使用早停机制")
    
    print("\n4.    可视化训练曲线:")
    print("   python visualize.py")
    
    print("\n5.    如果要继续训练:")
    print("   - 先备份当前最佳模型")
    print("   - 修改配置文件降低学习率")
    print("   - 使用 --resume 从最佳检查点继续")

def main():
    """主函数"""
    print(f"   分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    analyze_training_status()
    provide_recommendations()
    
    print(f"\n{'='*60}")
    print("分析完成")

if __name__ == "__main__":
    main()
