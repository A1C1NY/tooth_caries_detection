#!/usr/bin/env python3
"""
快速开始增量训练
一键运行，实时显示输出

用法：
python quick_start_training.py
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    print("   牙齿龋齿检测 - 增量训练快速开始")
    print("="*60)
    
    # 检查必要文件
    config_file = "config/config.yaml"
    train_script = "train.py"
    
    if not os.path.exists(config_file):
        print(f"   配置文件不存在: {config_file}")
        return 1
    
    if not os.path.exists(train_script):
        print(f"   训练脚本不存在: {train_script}")
        return 1
    
    print(f"   配置文件: {config_file}")
    print(f"   训练脚本: {train_script}")
    print()
    
    # 选择训练模式
    print("请选择训练模式:")
    print("1. 单次训练")
    print("2. 增量训练 (推荐)")
    print("3. 仅评估现有模型")
    
    while True:
        choice = input("\n请输入选择 (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("请输入有效选择 (1/2/3)")
    
    if choice == '1':
        # 单次训练
        print("\n   开始单次训练...")
        cmd = [sys.executable, "train.py", "--config", config_file]
        
    elif choice == '2':
        # 增量训练
        rounds = input("\n请输入训练轮数 (默认3): ").strip()
        if not rounds:
            rounds = "3"
        
        print(f"\n   开始{rounds}轮增量训练...")
        cmd = [sys.executable, "simple_incremental_training.py", 
               "--config", config_file, "--rounds", rounds]
        
    elif choice == '3':
        # 仅评估
        print("\n   开始模型评估...")
        cmd = [sys.executable, "train.py", "--config", config_file, "--eval-only"]
    
    # 显示即将执行的命令
    print(f"\n执行命令: {' '.join(cmd)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*60)
    
    try:
        # 运行命令
        result = subprocess.run(cmd)
        
        print("\n" + "="*60)
        if result.returncode == 0:
            print("   训练/评估成功完成!")
        else:
            print("   训练/评估失败!")
        
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n   训练被用户中断")
        return 1
    except Exception as e:
        print(f"\n   运行时发生错误: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        exit(1)
