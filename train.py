"""
PyTorch + torchvision Faster R-CNN 训练脚本
用于牙齿龋齿检测

使用 torchvision 内置的 Faster R-CNN 模型替代 Detectron2
保持与原配置文件完全兼容

"""

import os
import yaml
import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torchvision.ops import box_iou

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def setup_logging(log_dir):
    """设置日志记录系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return log_file

def load_config(config_path):
    """加载配置文件"""
    logger = logging.getLogger(__name__)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

class COCODataset(Dataset):
    """COCO格式数据集类"""
    
    def __init__(self, annotation_file, images_dir, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        # 加载COCO格式标注
        with open(annotation_file, 'r', encoding='utf-8') as f:  # 添加 encoding='utf-8'
            self.coco_data = json.load(f)
        
        # 创建索引映射
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # 按图像分组标注
        self.annotations_by_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        
        # 只保留有标注的图像
        self.image_ids = [img_id for img_id in self.images.keys() 
                         if img_id in self.annotations_by_image]
        
        print(f"数据集加载完成: {len(self.image_ids)} 张图像")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        annotations = self.annotations_by_image[image_id]
        
        # 加载图像
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # 准备目标数据
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO格式: [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # 转换为 [x1, y1, x2, y2] 格式
            boxes.append([x, y, x + w, y + h])
            
            # 类别ID (torchvision要求从1开始)
            labels.append(ann['category_id'])
        
        # 转换为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }
        
        if self.transforms:
            image = self.transforms(image)
        else:
            # 默认转换
            image = T.ToTensor()(image)
        
        return image, target

def get_transform(train=True):
    """获取数据变换"""
    transforms = []
    transforms.append(T.ToTensor())
    
    if train:
        # 训练时的数据增强
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

def create_model(num_classes, config):
    """创建Faster R-CNN模型"""
    logger = logging.getLogger(__name__)
    
    # 加载预训练的Faster R-CNN模型
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 设置设备
    device = torch.device(config['hardware']['device'])
    model.to(device)
    
    logger.info(f"模型创建完成，类别数: {num_classes}")
    logger.info(f"模型设备: {device}")
    
    return model

def collate_fn(batch):
    """数据加载器的collate函数"""
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, config):
    """训练一个epoch"""
    logger = logging.getLogger(__name__)
    model.train()
    
    running_loss = 0.0
    avg_loss = 0.0  # 添加这行：初始化 avg_loss
    log_period = config['training']['log_period']
    
    if len(data_loader) == 0:  # 添加这个检查
        logger.warning("数据加载器为空，跳过训练")
        return avg_loss
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        
        # 记录日志
        if (i + 1) % log_period == 0:
            avg_loss = running_loss / log_period
            logger.info(f"Epoch {epoch+1}, Step {i+1}/{len(data_loader)}, Loss: {avg_loss:.4f}")
            running_loss = 0.0
    
    # 如果最后还有剩余的loss没有计算平均值
    if running_loss > 0:
        remaining_steps = len(data_loader) % log_period
        if remaining_steps > 0:
            avg_loss = running_loss / remaining_steps
    
    return avg_loss

def evaluate_model(model, data_loader, device, config):
    """评估模型"""
    logger = logging.getLogger(__name__)
    model.eval()
    
    # 收集预测结果
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # 计算简单的评估指标
    conf_threshold = config['model']['roi_heads']['score_thresh_test']
    
    total_predictions = 0
    correct_predictions = 0
    
    for pred, target in zip(all_predictions, all_targets):
        # 过滤低置信度预测
        high_conf_mask = pred['scores'] > conf_threshold
        pred_boxes = pred['boxes'][high_conf_mask]
        pred_labels = pred['labels'][high_conf_mask]
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        total_predictions += len(pred_boxes)
        
        # 简单的IoU匹配
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            iou_matrix = box_iou(pred_boxes, target_boxes)
            max_ious, max_indices = iou_matrix.max(dim=1)
            
            # 计算正确预测（IoU > 0.5且类别正确）
            for i, max_iou in enumerate(max_ious):
                if max_iou > 0.5:
                    pred_label = pred_labels[i]
                    target_label = target_labels[max_indices[i]]
                    if pred_label == target_label:
                        correct_predictions += 1
    
    # 计算精度
    precision = correct_predictions / max(total_predictions, 1)
    
    logger.info(f"评估结果:")
    logger.info(f"  总预测数: {total_predictions}")
    logger.info(f"  正确预测数: {correct_predictions}")
    logger.info(f"  精度: {precision:.4f}")
    
    return {
        'precision': precision,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions
    }

def save_model_and_logs(model, config, log_file, epoch, metrics):
    """保存模型和日志"""
    logger = logging.getLogger(__name__)
    
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(output_dir, "model_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': config
    }, model_path)
    
    # 保存配置文件副本
    config_backup = os.path.join(output_dir, "config.yaml")
    with open(config_backup, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 复制日志文件
    log_backup = os.path.join(output_dir, os.path.basename(log_file))
    shutil.copy2(log_file, log_backup)
    
    logger.info(f"模型已保存到: {model_path}")
    logger.info(f"配置文件已备份到: {config_backup}")
    logger.info(f"日志文件已备份到: {log_backup}")

def print_system_info():
    """打印系统信息"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch未安装")
    
    logger.info(f"torchvision版本: {torchvision.__version__}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PyTorch牙齿龋齿检测模型训练")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否继续之前的训练"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="仅进行模型评估"
    )
    
    args = parser.parse_args()
    
    try:
        # 1. 加载配置
        config = load_config(args.config)
        
        # 2. 设置日志
        log_file = setup_logging(config['training']['log_dir'])
        logger = logging.getLogger(__name__)
        
        logger.info("="*60)
        logger.info("开始PyTorch牙齿龋齿检测模型训练")
        logger.info("="*60)
        
        # 3. 打印系统信息
        print_system_info()
        
        # 4. 设置设备
        device = torch.device(config['hardware']['device'])
        logger.info(f"使用设备: {device}")
        
        # 5. 创建数据集
        logger.info("创建数据集...")
        
        train_dataset = COCODataset(
            config['paths']['train_annotations'],
            config['data']['images_dir'],
            transforms=get_transform(train=True)
        )
        
        val_dataset = COCODataset(
            config['paths']['val_annotations'],
            config['data']['images_dir'],
            transforms=get_transform(train=False)
        )
        
        # 6. 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['dataloader']['batch_size'],
            shuffle=config['data']['dataloader']['shuffle'],
            num_workers=config['data']['dataloader']['num_workers'],
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['dataloader']['batch_size'],
            shuffle=False,
            num_workers=config['data']['dataloader']['num_workers'],
            collate_fn=collate_fn
        )
        
        # 7. 创建模型
        logger.info("创建模型...")
        model = create_model(config['data']['num_classes'] + 1, config)  # +1 for background
        
        # 8. 创建优化器
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['base_lr'],
            momentum=0.9,
            weight_decay=0.0005
        )
        
        # 9. 学习率调度器
        # 检查配置文件中是否有 steps 和 gamma 字段
        steps = config['training'].get('steps', [1000])  # 添加默认值
        gamma = config['training'].get('gamma', 0.1)     # 添加默认值
        
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=steps[0] // 100,  # 使用获取的值
            gamma=gamma                  # 使用获取的值
        )
        
        if args.eval_only:
            # 仅评估模式
            logger.info("运行评估模式...")
            if os.path.exists(config['inference']['model_path']):
                checkpoint = torch.load(config['inference']['model_path'])
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"加载模型: {config['inference']['model_path']}")
            
            metrics = evaluate_model(model, val_loader, device, config)
        else:
            # 训练模式
            logger.info("运行训练模式...")
            
            # 计算总epoch数（从max_iter转换）
            max_iter = config['training']['max_iter']
            total_epochs = max_iter // len(train_loader) + 1
            
            logger.info(f"训练参数:")
            logger.info(f"  总epochs: {total_epochs}")
            logger.info(f"  批次大小: {config['data']['dataloader']['batch_size']}")
            logger.info(f"  学习率: {config['training']['base_lr']}")
            
            best_precision = 0.0
            
            # 10. 训练循环
            for epoch in range(total_epochs):
                logger.info(f"开始第 {epoch+1}/{total_epochs} 个epoch")
                
                # 训练
                avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, config)
                
                # 更新学习率
                lr_scheduler.step()
                
                # 每隔几个epoch评估一次
                if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
                    logger.info("进行模型评估...")
                    metrics = evaluate_model(model, val_loader, device, config)
                    
                    # 保存最佳模型
                    if metrics['precision'] > best_precision:
                        best_precision = metrics['precision']
                        save_model_and_logs(model, config, log_file, epoch, metrics)
                        logger.info(f"保存新的最佳模型，精度: {best_precision:.4f}")
        
        logger.info("="*60)
        logger.info("训练/评估完成！")
        logger.info("="*60)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"训练过程中发生错误: {e}")
        logger.error("训练失败！")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())