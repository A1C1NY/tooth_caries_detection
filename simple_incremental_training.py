#!/usr/bin/env python3
"""
简化的增量训练脚本
实时显示训练输出，支持键盘中断

用法：
python simple_incremental_training.py --config config/config.yaml --rounds 3
"""

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime

def run_training_with_live_output(config_path, resume=False):
    """运行训练并实时显示输出"""
    cmd = [sys.executable, "train.py", "--config", config_path]
    if resume:
        cmd.append("--resume")
    
    print(f"\n{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        # 直接运行，输出会实时显示
        result = subprocess.run(cmd)
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n   用户中断训练 (Ctrl+C)")
        print("   训练状态已自动保存，可以使用 --resume 继续")
        return False
    except Exception as e:
        print(f"\n   运行训练时发生错误: {e}")
        return False

def check_improvement(output_dir):
    """检查训练改进情况"""
    checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")
    
    if os.path.exists(checkpoint_path):
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            metrics = checkpoint.get('metrics', {})
            epoch = checkpoint.get('epoch', 0)
            precision = metrics.get('precision', 0)
            
            print(f"\n{'='*50}")
            print(f"当前最佳性能:")
            print(f"  Epoch: {epoch + 1}")
            print(f"  精度: {precision:.4f}")
            print(f"  总预测数: {metrics.get('total_predictions', 0)}")
            print(f"  正确预测数: {metrics.get('correct_predictions', 0)}")
            print(f"{'='*50}\n")
            
            return precision
            
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return 0.0
    
    return 0.0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化增量训练脚本")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--rounds", type=int, default=3, help="训练轮数")
    parser.add_argument("--min-improvement", type=float, default=0.001, help="最小改进阈值")
    
    args = parser.parse_args()
    
    print("="*60)
    print("简化增量训练")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"训练轮数: {args.rounds}")
    print(f"最小改进阈值: {args.min_improvement}")
    print("\n按 Ctrl+C 可以随时中断训练")
    
    # 从配置文件获取输出目录
    try:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        output_dir = config['training']['output_dir']
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return 1
    
    best_precision = 0.0
    no_improvement_rounds = 0
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n   开始第 {round_num}/{args.rounds} 轮训练")
        print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 第一轮不使用resume，后续轮次使用resume
        resume = round_num > 1
        
        # 运行训练
        success = run_training_with_live_output(args.config, resume)
        
        if not success:
            print(f"\n   第 {round_num} 轮训练失败或被中断")
            break
        
        # 检查训练结果
        current_precision = check_improvement(output_dir)
        
        if round_num == 1:
            best_precision = current_precision
            print(f"   第 {round_num} 轮完成，基准精度: {current_precision:.4f}")
        else:
            improvement = current_precision - best_precision
            
            if improvement >= args.min_improvement:
                best_precision = current_precision
                no_improvement_rounds = 0
                print(f"   第 {round_num} 轮完成，精度提升: +{improvement:.4f} -> {current_precision:.4f}")
            else:
                no_improvement_rounds += 1
                print(f"    第 {round_num} 轮完成，改进较小: +{improvement:.4f} -> {current_precision:.4f}")
                
                if no_improvement_rounds >= 2:
                    print(f"\n   连续 {no_improvement_rounds} 轮无显著改进，建议停止训练")
                    user_input = input("是否继续训练？(y/N): ").strip().lower()
                    if user_input not in ['y', 'yes']:
                        print("用户选择停止训练")
                        break
        
        # 如果不是最后一轮，询问是否继续
        if round_num < args.rounds:
            print(f"\n    第 {round_num} 轮训练完成")
            user_input = input("按 Enter 继续下一轮训练，或输入 'q' 退出: ").strip().lower()
            if user_input == 'q':
                print("用户选择退出")
                break
    
    print(f"\n{'='*60}")
    print("增量训练完成")
    print(f"最终最佳精度: {best_precision:.4f}")
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        exit(1)
