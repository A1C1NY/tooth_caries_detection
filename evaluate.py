"""
详细的模型评估脚本
基于train.py扩展，提供更深入的分析

主要功能：
1. 多IoU阈值评估
2. 不同置信度阈值分析
3. 预测结果可视化
4. 错误案例分析
"""

import os
import yaml
import argparse
import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# 导入训练的类与函数
from train import COCODataset, create_model, collate_fn,  get_transform, load_config

def evaluate_model(model, data_loader, device, iou_thresholds = [0.3, 0.5, 0.7], conf_thresholds = [0.3, 0.5, 0.7]):
    """
    不同iou和置信度下的模型评估
    """
    print("Evaluating model...")
    model.eval()  # 设置模型为评估模式

    all_predictions = []
    all_targets = []

    with torch.no_grad(): # 评估模式下禁用梯度计算
        for i, (images, targets) in enumerate(data_loader):
            if i % 5 == 0:
                print(f"处理批次 { i + 1 } / { len(data_loader) }")

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    print("计算多阈值指标...")

    # 计算真实标注总数
    total_gt = sum(len(target['boxes']) for target in all_targets)  

    results = {}

    for conf_th in conf_thresholds:
        results[conf_th] = {}
        
        for iou_th in iou_thresholds:
            tp = 0  # True Positives
            fp = 0  # False Positives
            matched_gt = set()  # 已匹配的真实框
            
            total_predictions = 0
            
            for pred_idx, (pred, target) in enumerate(zip(all_predictions, all_targets)):
                # 过滤低置信度预测
                """
                如何过滤？
                通过检查预测的置信度分数是否高于设定的阈值
                只有高于阈值的预测才会被保留用于后续计算

                """
                high_conf_mask = pred['scores'] > conf_th
                pred_boxes = pred['boxes'][high_conf_mask]
                pred_labels = pred['labels'][high_conf_mask]
                pred_scores = pred['scores'][high_conf_mask]
                
                target_boxes = target['boxes']
                target_labels = target['labels']
                
                total_predictions += len(pred_boxes) # 预测框总数
                
                # 对每个预测框找最佳匹配
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(target_boxes, target_labels)):
                        gt_key = (pred_idx, gt_idx)
                        if gt_key in matched_gt:
                            continue
                        
                        if pred_label == gt_label:
                            # 计算IoU
                            """
                            计算原理：
                            IoU = 交集面积 / 并集面积
                            交集面积 = max(0, x2 - x1) * max(0, y2 - y1)
                            并集面积 = pred_area + gt_area - 交集面积

                            """
                            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) # 预测框面积
                            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) # 真实框面积
                            
                            """
                            下面的计算方式是基于预测框和真实框的坐标来计算交集面积
                            通过取最大和最小值来确定交集区域的坐标
                            比如：
                            pred_box = [x1, y1, x2, y2]
                            gt_box = [x1_gt, y1_gt, x2_gt, y2_gt]
                            交集区域的左上角坐标是 (max(x1, x1_gt), max(y1, y1_gt))
                            交集区域的右下角坐标是 (min(x2, x2_gt), min(y2, y2_gt))
                            如果交集区域的宽度和高度都大于0，则说明有交集
                            
                            """
                            x1 = max(pred_box[0], gt_box[0])
                            y1 = max(pred_box[1], gt_box[1])
                            x2 = min(pred_box[2], gt_box[2])
                            y2 = min(pred_box[3], gt_box[3])
                            
                            if x2 > x1 and y2 > y1:
                                intersection = (x2 - x1) * (y2 - y1) # 交集面积
                                union = pred_area + gt_area - intersection # 并集面积
                                iou = intersection / union if union > 0 else 0
                                
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = gt_idx # 记录最佳匹配的真实框索引
                    
                    if best_iou >= iou_th and best_gt_idx >= 0:
                        tp += 1 # 预测正确
                        matched_gt.add((pred_idx, best_gt_idx))
                    else:
                        fp += 1 # 预测错误或未匹配
            
            fn = total_gt - len(matched_gt) # 未匹配的真实框数量
            
            '''
            计算指标：
            precision = TP / (TP + FP)，表示预测正确的比例
            recall = TP / (TP + FN)，表示真实框被正确预测的比例
            f1_score = 2 * (precision * recall) / (precision + recall)，综合考虑精确率和召回率的指标

            result包括：
            precision, recall, f1_score, tp, fp, fn, total_predictions, total_ground_truth
            total_predictions = 预测框总数
            total_ground_truth = 真实框总数
            tp = 预测正确的数量
            fp = 预测错误的数量
            fn = 未匹配的真实框数量

            '''
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0 
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[conf_th][iou_th] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'total_predictions': total_predictions,
                'total_ground_truth': total_gt
            }
    
    return results


def create_analysis_plots(results, output_dir):
    """创建分析图表"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    conf_thresholds = list(results.keys())
    iou_thresholds = list(results[conf_thresholds[0]].keys())
    
    # 1. Precision vs IoU
    ax = axes[0, 0]
    for conf_th in conf_thresholds:
        precisions = [results[conf_th][iou]['precision'] for iou in iou_thresholds]
        ax.plot(iou_thresholds, precisions, marker='o', label=f'Conf={conf_th}')
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs IoU Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Recall vs IoU
    ax = axes[0, 1]
    for conf_th in conf_thresholds:
        recalls = [results[conf_th][iou]['recall'] for iou in iou_thresholds]
        ax.plot(iou_thresholds, recalls, marker='s', label=f'Conf={conf_th}')
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs IoU Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1 Score vs IoU
    ax = axes[0, 2]
    for conf_th in conf_thresholds:
        f1_scores = [results[conf_th][iou]['f1_score'] for iou in iou_thresholds]
        ax.plot(iou_thresholds, f1_scores, marker='^', label=f'Conf={conf_th}')
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs IoU Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Precision-Recall曲线
    ax = axes[1, 0]
    for conf_th in conf_thresholds:
        precisions = [results[conf_th][iou]['precision'] for iou in iou_thresholds]
        recalls = [results[conf_th][iou]['recall'] for iou in iou_thresholds]
        ax.plot(recalls, precisions, marker='o', label=f'Conf={conf_th}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 预测数量统计
    ax = axes[1, 1]
    conf_th = 0.5  # 使用标准置信度
    tps = [results[conf_th][iou]['tp'] for iou in iou_thresholds]
    fps = [results[conf_th][iou]['fp'] for iou in iou_thresholds]
    fns = [results[conf_th][iou]['fn'] for iou in iou_thresholds]
    
    x = np.arange(len(iou_thresholds))
    width = 0.25
    
    ax.bar(x - width, tps, width, label='True Positives', alpha=0.8, color='green')
    ax.bar(x, fps, width, label='False Positives', alpha=0.8, color='red')
    ax.bar(x + width, fns, width, label='False Negatives', alpha=0.8, color='orange')
    
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('Count')
    ax.set_title(f'Detection Statistics (Conf={conf_th})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{iou:.1f}' for iou in iou_thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 最佳性能总结
    ax = axes[1, 2]
    best_f1 = 0
    best_conf = 0
    best_iou = 0
    
    for conf_th in results:
        for iou_th in results[conf_th]:
            f1 = results[conf_th][iou_th]['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_conf = conf_th
                best_iou = iou_th
    
    best_metrics = results[best_conf][best_iou]
    
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [best_metrics['precision'], best_metrics['recall'], best_metrics['f1_score']]
    
    bars = ax.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel('Score')
    ax.set_title(f'Best Performance\n(Conf={best_conf}, IoU={best_iou})')
    ax.set_ylim(0, 1)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_dir, 'detailed_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   分析图表已保存: {plot_path}")
    
    plt.show()
    
    return best_conf, best_iou, best_f1

def print_detailed_report(results, output_dir):
    """打印详细报告"""
    
    report_path = os.path.join(output_dir, 'detailed_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("   牙齿龋齿检测模型 - 详细评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 找到最佳性能
        best_f1 = 0
        best_conf = 0
        best_iou = 0
        
        for conf_th in results:
            for iou_th in results[conf_th]:
                f1 = results[conf_th][iou_th]['f1_score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_conf = conf_th
                    best_iou = iou_th
        
        f.write(f"   最佳性能:\n")
        f.write(f"   置信度阈值: {best_conf}\n")
        f.write(f"   IoU阈值: {best_iou}\n")
        f.write(f"   F1分数: {best_f1:.4f}\n")
        f.write(f"   精度: {results[best_conf][best_iou]['precision']:.4f}\n")
        f.write(f"   召回率: {results[best_conf][best_iou]['recall']:.4f}\n\n")
        
        # 详细结果表格
        for conf_th in results:
            f.write(f"  置信度阈值 = {conf_th}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'IoU':<6} {'Precision':<10} {'Recall':<8} {'F1':<8} {'TP':<4} {'FP':<4} {'FN':<4}\n")
            f.write("-" * 50 + "\n")
            
            for iou_th in results[conf_th]:
                metrics = results[conf_th][iou_th]
                f.write(f"{iou_th:<6.1f} {metrics['precision']:<10.3f} {metrics['recall']:<8.3f} "
                       f"{metrics['f1_score']:<8.3f} {metrics['tp']:<4} {metrics['fp']:<4} {metrics['fn']:<4}\n")
            f.write("\n")
    
    print(f" 详细报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="详细模型评估脚本")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help="输出目录")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    print("加载配置结束")

    print("开始评估")

    device = torch.device(config['hardware']['device'])


    val_dataset = COCODataset(
        config['paths']['val_annotations'],
        config['data']['images_dir'],
        transforms = get_transform(train=False),
        
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['data']['dataloader']['num_workers'],
        collate_fn=collate_fn
    )

    print("验证集加载完成")

    # 创建和加载模型
    model = create_model(config['data']['num_classes'] + 1, config)
    
    # 加载训练好的权重
    if os.path.exists(config['inference']['model_path']):
        checkpoint = torch.load(config['inference']['model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   模型加载完成: {config['inference']['model_path']}")
    else:
        print("   未找到训练好的模型!")
        return
    
    # 多阈值评估
    results = evaluate_model(
        model, val_loader, device,
        iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        conf_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成分析图表
    best_conf, best_iou, best_f1 = create_analysis_plots(results, args.output_dir)
    
    # 生成详细报告
    print_detailed_report(results, args.output_dir)
    
    print(f"\n   详细评估完成!")
    print(f"   最佳F1分数: {best_f1:.4f} (Conf={best_conf}, IoU={best_iou})")
    print(f"   结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()