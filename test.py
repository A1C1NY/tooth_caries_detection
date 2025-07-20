"""
测试脚本

"""

import os
import torch
import json
import logging
from torch.utils.data import DataLoader
from train import COCODataset, create_model, collate_fn, get_transform, load_config
from evaluate import calculate_detection_metrics, save_evaluation_report


def test_model(model, test_loader, device, config):
    logger = logging.getLogger(__name__)
    logger.info("开始测试模型...")

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            all_predictions.extend(predictions)