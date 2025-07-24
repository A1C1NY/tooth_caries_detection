#!/usr/bin/env python3
"""
模型状态检查工具
检查训练是否已安全保存，以及最佳模型状态
"""

import os
import torch
import yaml
from datetime import datetime

def check_model_safety():
    """检查模型是否已安全保存"""
    print("   模型安全性检查")
    print("="*60)
    
    # 检查配置文件
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print("   配置文件不存在")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['training']['output_dir']
    print(f"   输出目录: {output_dir}")
    
    # 检查关键文件
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    best_checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")
    model_final_path = os.path.join(output_dir, "model_final.pth")
    config_backup_path = os.path.join(output_dir, "config.yaml")
    
    files_to_check = [
        ("当前检查点", checkpoint_path),
        ("最佳检查点", best_checkpoint_path),
        ("最终模型", model_final_path),
        ("配置备份", config_backup_path)
    ]
    
    safety_score = 0
    total_files = len(files_to_check)
    
    print(f"\n   文件完整性检查:")
    for name, path in files_to_check:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"     {name}: 存在 ({file_size:.1f} MB)")
            safety_score += 1
        else:
            print(f"     {name}: 不存在")
    
    print(f"\n   安全性评分: {safety_score}/{total_files}")
    
    # 检查最佳模型详情
    if os.path.exists(best_checkpoint_path):
        print(f"\n🏆 最佳模型详情:")
        try:
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            metrics = checkpoint.get('metrics', {})
            precision = metrics.get('precision', 0)
            
            print(f"     Epoch: {epoch + 1}")
            print(f"     精度: {precision:.4f}")
            print(f"     总预测数: {metrics.get('total_predictions', 0)}")
            print(f"     正确预测数: {metrics.get('correct_predictions', 0)}")
            
            # 检查模型状态
            if 'model_state_dict' in checkpoint:
                print(f"     模型权重: 已保存")
            if 'optimizer_state_dict' in checkpoint:
                print(f"      优化器状态: 已保存")
            if 'lr_scheduler_state_dict' in checkpoint:
                print(f"     学习率调度器: 已保存")
                
        except Exception as e:
            print(f"     检查点损坏: {e}")
            safety_score -= 1
    
    # 检查当前检查点
    if os.path.exists(checkpoint_path):
        print(f"\n   当前检查点详情:")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            metrics = checkpoint.get('metrics', {})
            precision = metrics.get('precision', 0)
            
            print(f"     Epoch: {epoch + 1}")
            print(f"     精度: {precision:.4f}")
            
        except Exception as e:
            print(f"     当前检查点损坏: {e}")
    
    # 安全性评估
    print(f"\n    安全性评估:")
    if safety_score >= 3:
        print("     模型状态安全，可以放心停止训练")
        print("     建议：使用最佳模型进行后续操作")
    elif safety_score >= 2:
        print("      模型部分安全，建议备份现有文件")
        print("     建议：先保存当前状态再停止")
    else:
        print("     模型状态不安全！")
        print("     建议：不要停止训练，等待下一次评估保存")
    
    return safety_score >= 2

def provide_safe_stop_guide():
    """提供安全停止指南"""
    print(f"\n📋 安全停止训练指南:")
    print("="*50)
    
    print("   如果训练正在运行且想要安全停止:")
    print("1. 等待当前 epoch 完成")
    print("2. 在看到 '开始第 X 个epoch' 时按 Ctrl+C")
    print("3. 训练会自动保存当前状态并安全退出")
    
    print(f"\n   停止后的操作选项:")
    print("1. 评估最佳模型:")
    print("   python train.py --config config/config.yaml --eval-only")
    
    print("\n2. 继续训练 (从最佳检查点):")
    print("   python train.py --config config/config.yaml --resume")
    
    print("\n3. 分析训练状态:")
    print("   python analyze_training.py")
    
    print("\n4. 应急处理 (如果出现问题):")
    print("   python emergency_training_handler.py")

def check_if_training_running():
    """检查是否有训练进程在运行"""
    import psutil
    
    print(f"\n   进程检查:")
    training_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('train.py' in arg for arg in cmdline):
                training_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if training_processes:
        print("     检测到训练进程正在运行:")
        for proc in training_processes:
            print(f"    PID: {proc['pid']}, 命令: {' '.join(proc['cmdline'][:3])}...")
        print("     可以使用 Ctrl+C 安全停止")
    else:
        print("     没有检测到训练进程")
    
    return len(training_processes) > 0

def main():
    """主函数"""
    print(f"   检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查训练进程
    is_training = check_if_training_running()
    
    # 检查模型安全性
    is_safe = check_model_safety()
    
    # 提供指南
    provide_safe_stop_guide()
    
    print(f"\n{'='*60}")
    if is_safe:
        print("   模型状态安全，可以放心操作")
    else:
        print("    建议谨慎操作，确保模型安全")
    
    if is_training:
        print("   训练正在进行中")
    else:
        print("   训练已停止")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if 'psutil' in str(e):
            print("   需要安装 psutil: pip install psutil")
            # 不检查进程，直接检查模型
            print(f"   检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            check_model_safety()
            provide_safe_stop_guide()
        else:
            raise
