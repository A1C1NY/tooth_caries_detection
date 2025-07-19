"""
将LABELME文件转换为COCO格式的脚本。
用于Detectron2训练。

"""

import os
import json
import yaml
import argparse
from pathlib import Path
from src.data.convert_labelme import LabelMeToCOCOConverter

def load_config(config_path):
    """
    加载本地配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # parser是命令行参数解析器
    # 用于解析命令行参数    
    parser = argparse.ArgumentParser(description='Convert LabelMe annotations to COCO format')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)
    print("Loaded configuration")

    # 创建转换器
    converter = LabelMeToCOCOConverter(config, verbose = args.verbose)

    # 转换
    print("Starting conversion...")
    converter.convert()
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()