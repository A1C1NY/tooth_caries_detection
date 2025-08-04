'''
预测结果可视化
将预测框和真实框绘制在同一张图上并保存

'''

import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train import COCODataset, create_model, collate_fn, get_transform, load_config

def compute_image_score(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh=0.5):
    """计算单张图片的 TP, FP, FN 及 score"""
    from torchvision.ops import box_iou
    import numpy as np
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, 1.0  # 没有目标，视为完美
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), 0.0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0.0
    iou_matrix = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes))
    matched_gt = set()
    tp = 0
    fp = 0
    for i, pred_label in enumerate(pred_labels):
        ious = iou_matrix[i]
        max_iou, max_idx = ious.max(0)
        if max_iou > iou_thresh and pred_label == gt_labels[max_idx] and max_idx.item() not in matched_gt:
            tp += 1
            matched_gt.add(max_idx.item())
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    denom = tp + fp + fn
    score = tp / denom if denom > 0 else 0.0
    return tp, fp, fn, score


def create_mosaic(image_paths, out_path, title):
    """将16张图片拼成4x4大图"""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for idx, img_path in enumerate(image_paths):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(img_path), fontsize=8)
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    plt.close()


def visualize_and_save(model, data_loader, device, save_dir, val_dataset, config, max_images=None):
    """
    可视化并保存图片，并生成top/bottom16拼图
    """
    
    if max_images is None:
        print(f"   开始可视化所有验证集图片，保存到: {save_dir}")
    else:
        print(f"   开始可视化前{max_images}张图片，保存到: {save_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0
    image_scores = []  # 新增：记录每张图片的score和路径
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if max_images is not None and count >= max_images:
                break
                
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)
            
            for i, (image, pred, target) in enumerate(zip(images, predictions, targets)):
                if max_images is not None and count >= max_images:
                    break
                
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).cpu().numpy()
                    if image.max() <= 1.0:
                        image = (image * 255).astype('uint8') / 255.0
                
                # 获取原图文件名 - 根据实际数据集结构修正
                try:
                    # 方法1：从target获取image_id
                    img_id = target.get('image_id', None)
                    if isinstance(img_id, torch.Tensor):
                        img_id = img_id.item()
                    
                    # 根据你的数据集结构调整
                    if img_id is not None and hasattr(val_dataset, 'coco_data'):
                        # 在coco_data中查找图片信息
                        images_info = val_dataset.coco_data.get('images', [])
                        for img_info in images_info:
                            if img_info['id'] == img_id:
                                original_filename = img_info['file_name']
                                break
                        else:
                            original_filename = f'id_{img_id}_not_found'
                    elif hasattr(val_dataset, 'images') and hasattr(val_dataset, 'image_ids'):
                        # 方法2：从数据集索引直接获取
                        current_idx = batch_idx * data_loader.batch_size + i
                        if current_idx < len(val_dataset.image_ids):
                            dataset_img_id = val_dataset.image_ids[current_idx]
                            # 在images中查找对应的文件名
                            for img_info in val_dataset.images:
                                if img_info['id'] == dataset_img_id:
                                    original_filename = img_info['file_name']
                                    break
                            else:
                                original_filename = f'idx_{current_idx}_id_{dataset_img_id}'
                        else:
                            original_filename = f'unknown_{count+1}'
                    else:
                        original_filename = f'unknown_{count+1}'
                except Exception as e:
                    original_filename = f'error_{count+1}'
                
                # 创建包含两种标号的文件名
                image_number = f'image_{count+1}'
                filename_for_title = original_filename
                
                print(f"   处理图片: {original_filename}")
                
                # 获取置信度阈值
                score_threshold = config.get('inference', {}).get('score_threshold', 0.5)
                
                # 创建图片
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(image)
                ax.set_title(f'File: {filename_for_title}\nGreen: Ground Truth, Red: Predictions (Score>{score_threshold})', 
                           fontsize=14, fontweight='bold')
                
                # 绘制真实框 (绿色)
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    width, height = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=3, edgecolor='green', facecolor='none')
                    ax.add_patch(rect)
                
                # 绘制预测框 (红色，使用配置文件中的阈值)
                high_conf_mask = pred['scores'] > score_threshold
                pred_boxes = pred['boxes'][high_conf_mask].cpu().numpy()
                pred_labels = pred['labels'][high_conf_mask].cpu().numpy()
                pred_scores = pred['scores'][high_conf_mask].cpu().numpy()
                
                for box, score in zip(pred_boxes, pred_scores):
                    x1, y1, x2, y2 = box
                    width, height = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=2, edgecolor='red', 
                                           facecolor='red', alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{score:.2f}', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8),
                           fontsize=9, color='white', weight='bold')
                
                ax.axis('off')
                
                # 保存图片 - 包含两种标号
                save_path = os.path.join(save_dir, f'{image_number}_{original_filename}_prediction.png')
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()
                # 新增：计算score并记录
                tp, fp, fn, score = compute_image_score(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh=0.5)
                image_scores.append({
                    'score': score,
                    'path': save_path,
                    'filename': original_filename,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                })
                count += 1
                
                if count % 5 == 0:
                    print(f"   已处理 {count} 张图片...")
    
    print(f"\n   完成！共保存 {count} 张可视化图片")
    # 新增：生成top16和bottom16拼图
    if len(image_scores) >= 16:
        image_scores_sorted = sorted(image_scores, key=lambda x: x['score'], reverse=True)
        top16 = [x['path'] for x in image_scores_sorted[:16]]
        bottom16 = [x['path'] for x in image_scores_sorted[-16:]]
        create_mosaic(top16, os.path.join(save_dir, 'top16_accuracy.png'), 'Top 16 Accuracy Images')
        create_mosaic(bottom16, os.path.join(save_dir, 'bottom16_accuracy.png'), 'Bottom 16 Accuracy Images')
        print(f"\n   已生成top16和bottom16拼图！")

def main():
    # 加载配置 - 使用训练时的实际配置文件
    config = load_config('experiments/tooth_caries_model/config.yaml')
    device = torch.device(config['hardware']['device'])
    print(f"使用设备: {device}")
    print(f"使用配置文件: experiments/tooth_caries_model/config.yaml")
    
    # 创建模型并加载权重
    model = create_model(config['data']['num_classes'] + 1, config)
    
    if os.path.exists(config['inference']['model_path']):
        checkpoint = torch.load(config['inference']['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"   模型加载完成: {config['inference']['model_path']}")
    else:
        print("   未找到训练好的模型!")
        return
    
    # 创建验证集 
    val_dataset = COCODataset(
        config['paths']['val_annotations'],
        config['data']['images_dir'],
        transforms=get_transform(train=False)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2,
        shuffle=False, 
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"   验证集加载完成，共 {len(val_dataset)} 张图片")
    
    save_directory = 'data/visualizations'
    visualize_and_save(model, val_loader, device, save_directory, val_dataset, config, max_images=None)
    
    print(f"   可视化图片已保存到: {save_directory}")

if __name__ == "__main__":
    main()