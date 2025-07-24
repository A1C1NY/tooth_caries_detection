#!/usr/bin/env python3
"""
训练应急处理脚本
当检测到过拟合时的应急操作
"""

import os
import sys
import torch
import yaml
import shutil
from datetime import datetime

def emergency_stop_training():
    """应急停止训练并保存状态"""
    print("   检测到训练异常，执行应急处理...")
    print("="*60)
    
    # 1. 检查配置
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print("   配置文件不存在")
        return 1
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['training']['output_dir']
    
    # 2. 备份当前状态
    backup_dir = f"emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(output_dir):
        shutil.copytree(output_dir, backup_dir)
        print(f"   已备份训练状态到: {backup_dir}")
    
    # 3. 分析检查点
    best_checkpoint = os.path.join(output_dir, "checkpoint_best.pth")
    current_checkpoint = os.path.join(output_dir, "checkpoint.pth")
    
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        best_metrics = checkpoint.get('metrics', {})
        best_epoch = checkpoint.get('epoch', 0)
        
        print(f"\n   最佳模型状态:")
        print(f"  Epoch: {best_epoch + 1}")
        print(f"  精度: {best_metrics.get('precision', 0):.4f}")
        print(f"  总预测数: {best_metrics.get('total_predictions', 0)}")
        print(f"  正确预测数: {best_metrics.get('correct_predictions', 0)}")
        
        # 4. 将最佳模型设为当前模型
        if os.path.exists(current_checkpoint):
            shutil.copy2(best_checkpoint, current_checkpoint)
            print(f"   已恢复到最佳检查点")
    
    # 5. 生成新的配置文件（降低学习率）
    new_config = config.copy()
    original_lr = config['training']['base_lr']
    new_lr = original_lr * 0.1  # 学习率降低10倍
    new_config['training']['base_lr'] = new_lr
    
    new_config_path = f"config/config_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(new_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n   生成应急配置文件: {new_config_path}")
    print(f"  学习率已调整: {original_lr} -> {new_lr}")
    
    return 0

def check_training_health():
    """检查训练健康状态"""
    print("   训练健康检查")
    print("="*60)
    
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print("   配置文件不存在")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['training']['output_dir']
    best_checkpoint = os.path.join(output_dir, "checkpoint_best.pth")
    current_checkpoint = os.path.join(output_dir, "checkpoint.pth")
    
    if not os.path.exists(best_checkpoint) or not os.path.exists(current_checkpoint):
        print("   检查点文件不完整")
        return False
    
    try:
        best = torch.load(best_checkpoint, map_location='cpu')
        current = torch.load(current_checkpoint, map_location='cpu')
        
        best_precision = best.get('metrics', {}).get('precision', 0)
        current_precision = current.get('metrics', {}).get('precision', 0)
        
        precision_diff = current_precision - best_precision
        
        print(f"最佳精度: {best_precision:.4f}")
        print(f"当前精度: {current_precision:.4f}")
        print(f"精度差异: {precision_diff:.4f}")
        
        if precision_diff < -0.01:  # 下降超过1%
            print("   检测到性能严重下降！")
            return False
        elif precision_diff < -0.005:  # 下降超过0.5%
            print("    检测到性能轻微下降")
            return False
        else:
            print("   训练状态正常")
            return True
            
    except Exception as e:
        print(f"   检查过程出错: {e}")
        return False

def provide_next_steps():
    """提供后续操作建议"""
    print(f"\n   后续操作建议:")
    print("="*50)
    
    print("1.    确认训练已停止")
    print("   - 检查是否还有 python train.py 进程在运行")
    print("   - 如果有，使用 Ctrl+C 或任务管理器停止")
    
    print("\n2.    评估最佳模型:")
    print("   python train.py --config config/config.yaml --eval-only")
    
    print("\n3.    使用应急配置继续训练:")
    print("   python train.py --config config/config_emergency_*.yaml --resume")
    
    print("\n4.    可视化分析:")
    print("   python analyze_training.py")
    
    print("\n5.    如果仍要继续原配置:")
    print("   - 先手动修改 config.yaml 中的 base_lr")
    print("   - 建议设为当前值的 0.1 倍")
    print("   - 然后使用 --resume 继续训练")

def main():
    """主函数"""
    print("   训练应急处理工具")
    print(f"   处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 检查训练健康状态
    is_healthy = check_training_health()
    
    if not is_healthy:
        print("\n执行应急处理...")
        result = emergency_stop_training()
        if result == 0:
            print("\n   应急处理完成")
        else:
            print("\n   应急处理失败")
    
    provide_next_steps()
    
    print(f"\n{'='*60}")
    print("应急处理完成")

if __name__ == "__main__":
    main()
