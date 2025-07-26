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
import yaml
import torch

def parse_training_logs(output_dir):
    """解析训练日志，找到最佳精度"""
    import glob
    import re
    
    # 查找所有训练日志
    log_files = glob.glob(os.path.join(output_dir, "training_*.log"))
    if not log_files:
        return None
    
    # 按文件名排序，获取最新的日志
    log_files.sort()
    
    # 首先尝试从现有的最佳checkpoint读取历史最佳精度作为基准
    best_checkpoint = os.path.join(output_dir, "checkpoint_best.pth")
    historical_best = 0
    if os.path.exists(best_checkpoint):
        try:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            metrics = checkpoint.get('metrics', {})
            historical_best = metrics.get('precision', 0)
            print(f"   读取到历史最佳精度: {historical_best:.4f}")
        except:
            pass
    
    best_precision = historical_best  # 从历史最佳开始，而不是从0开始
    best_epoch = 0
    best_total = 0
    best_correct = 0
    
    # 从最新的几个日志文件中查找是否有更好的精度
    for log_file in log_files[-3:]:  # 检查最新的3个日志文件
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 避免中文乱码，直接搜索数字模式和关键英文词
            lines = content.split('\n')
            epoch_matches = re.findall(r'Epoch\s+(\d+)', content)
            
            # 查找评估结果的行（通常包含特定的数字模式）
            for line in lines:
                # 匹配形如 "总预测数: 1514" 和 "正确预测数: 1089" 的行
                if ':' in line and re.search(r'\d{3,}', line):
                    numbers = re.findall(r'(\d+)', line)
                    if len(numbers) >= 2:
                        # 假设第一个大数字是总预测数，第二个是正确预测数
                        total_num = int(numbers[0])
                        correct_num = int(numbers[1])
                        if total_num > 100 and correct_num <= total_num:  # 合理性检查
                            current_precision = correct_num / total_num
                            # 只有超过历史最佳时才更新
                            if current_precision > best_precision:
                                print(f"   发现更好的精度: {current_precision:.4f} > {best_precision:.4f}")
                                best_precision = current_precision
                                best_total = total_num
                                best_correct = correct_num
                                if epoch_matches:
                                    best_epoch = int(epoch_matches[-1])
                
                # 也尝试直接匹配精度值（0.xxxx格式）
                precision_in_line = re.findall(r'([0-9]\.[0-9]{3,4})', line)
                for prec_str in precision_in_line:
                    prec = float(prec_str)
                    # 只有超过历史最佳且在合理范围内才更新
                    if 0.1 <= prec <= 1.0 and prec > best_precision:
                        print(f"   发现更好的精度: {prec:.4f} > {best_precision:.4f}")
                        best_precision = prec
                        if epoch_matches:
                            best_epoch = int(epoch_matches[-1])
                        
        except Exception as e:
            continue
    
    # 如果找到了有效的精度（包括历史最佳），返回结果
    if best_precision > 0:
        return {
            'precision': best_precision,
            'epoch': best_epoch,
            'total_predictions': best_total,
            'correct_predictions': best_correct,
            'is_historical': best_precision == historical_best  # 标记是否为历史最佳
        }
    return None

def check_existing_model():
    """检查现有模型状态"""
    try:
        # 加载配置
        config_file = "config/config.yaml"
        if not os.path.exists(config_file):
            return None, None
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        output_dir = config['training']['output_dir']
        
        # 检查所有可能的模型文件
        model_files = []
        for model_name in ["checkpoint_best.pth", "model_final.pth", "checkpoint.pth"]:
            model_path = os.path.join(output_dir, model_name)
            if os.path.exists(model_path):
                model_files.append((model_name, model_path))
        
        if not model_files:
            return config, None
        
        # 从训练日志中解析最佳结果
        log_info = parse_training_logs(output_dir)
        
        if log_info:
            # 使用找到的第一个模型文件
            checkpoint_path = model_files[0][1]
            
            model_info = {
                'epoch': log_info['epoch'],
                'precision': log_info['precision'],
                'total_predictions': log_info['total_predictions'],
                'correct_predictions': log_info['correct_predictions'],
                'checkpoint_path': checkpoint_path,
                'available_models': [f[0] for f in model_files],
                'is_historical': log_info.get('is_historical', False)
            }
            return config, model_info
        else:
            # 如果无法从日志解析，尝试从checkpoint中读取
            try:
                checkpoint_path = model_files[0][1]
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                metrics = checkpoint.get('metrics', {})
                
                model_info = {
                    'epoch': checkpoint.get('epoch', 0),
                    'precision': metrics.get('precision', 0),
                    'total_predictions': metrics.get('total_predictions', 0),
                    'correct_predictions': metrics.get('correct_predictions', 0),
                    'checkpoint_path': checkpoint_path,
                    'available_models': [f[0] for f in model_files],
                    'is_historical': True  # 从checkpoint读取的都是历史数据
                }
                return config, model_info
            except:
                return config, None
            
    except Exception as e:
        print(f"   检查现有模型时出错: {e}")
        return None, None

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
    
    # 检查现有模型状态
    print("\n" + "-"*40)
    print("   现有模型状态检查")
    print("-"*40)
    
    config, model_info = check_existing_model()
    
    if model_info:
        print(f"      发现已训练模型:")
        print(f"     - 训练轮次: {model_info['epoch']}")
        print(f"     - 当前精度: {model_info['precision']:.4f}")
        print(f"     - 总预测数: {model_info['total_predictions']}")
        print(f"     - 正确预测: {model_info['correct_predictions']}")
        print(f"     - 模型路径: {model_info['checkpoint_path']}")
        
        # 显示精度来源
        if model_info.get('is_historical', False):
            print(f"     - 精度来源: 历史最佳模型")
        else:
            print(f"     - 精度来源: 最新训练日志")
        
        if model_info['precision'] > 0.7:
            print(f"      模型性能良好 (精度 > 0.7)! 建议继续增量训练或直接评估")
        elif model_info['precision'] > 0.6:
            print(f"      模型性能中等，建议继续训练优化")
        else:
            print(f"      模型需要进一步训练")
    else:
        print("      未发现已训练模型，建议从头开始训练")
    
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
