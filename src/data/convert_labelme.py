"""
Change the LABELME file to COCO format script.
Treating the rectangle to COCO format.

"""

import os
import json
import cv2
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import random
# tqdm 用于显示进度条
from tqdm import tqdm

class LabelMeToCOCOConverter:
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.categories = []
        self.category_name_to_id = {}
        self.annotation_id = 1  # 用于生成唯一的注释ID
        self.image_id = 1  # 用于生成唯一的图片ID

        self._init_categories()

    def _init_categories(self):
        """初始化类别信息"""
        for idx, category_name in enumerate(self.config['data']['categories'], 1):
            category = {
                "id": idx,
                "name": category_name,
                "supercategory": "dental"  # 牙科相关的超类别
            }
            self.categories.append(category)
            self.category_name_to_id[category_name] = idx
            
        print(f"初始化了 {len(self.categories)} 个类别: {list(self.category_name_to_id.keys())}")
    
    def collect_files(self, images_dir, labelme_dir):
        """
        收集图片和对应的标注文件
        
        Args:
            images_dir (str): 图片目录路径
            labelme_dir (str): LabelMe标注文件目录路径
            
        Returns:
            list: 匹配的文件对列表 [(image_path, json_path), ...]
        """
        images_dir = Path(images_dir)
        labelme_dir = Path(labelme_dir)
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # 收集所有图片文件
        image_files = {}
        for ext in image_extensions:
            for img_path in images_dir.glob(f"*{ext}"): # 支持小写扩展名
                base_name = img_path.stem
                image_files[base_name] = img_path
            for img_path in images_dir.glob(f"*{ext.upper()}"): # 支持大写扩展名
                base_name = img_path.stem
                image_files[base_name] = img_path
        
        # 收集标注文件并匹配
        matched_files = []
        for json_path in labelme_dir.glob("*.json"):
            base_name = json_path.stem
            if base_name in image_files:
                matched_files.append((image_files[base_name], json_path))
        
        print(f"找到 {len(image_files)} 个图片文件")
        print(f"找到 {len(list(labelme_dir.glob('*.json')))} 个标注文件")
        print(f"成功匹配 {len(matched_files)} 对图片-标注文件")
        
        return matched_files
    
    def split_dataset(self, file_pairs, train_ratio=0.8, val_ratio=0.2):
        """
        将数据集分割为训练集和验证集（合并原来的验证集和测试集）
        
        Args:
            file_pairs (list): 匹配的文件对列表 [(image_path, json_path), ...]
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例（原验证集+测试集）
            
        Returns:
            dict: {"train": train_files, "val": val_files}
        """
        random.shuffle(file_pairs)

        total_files = len(file_pairs)
        train_count = int(total_files * train_ratio)
        
        # 验证集现在包含原来的验证集和测试集
        splits = {
            "train": file_pairs[:train_count],
            "val": file_pairs[train_count:]  # 剩余的20%全部作为验证集
        }

        print(f"数据集划分:")
        print(f"   训练集: {len(splits['train'])} 个文件 ({len(splits['train'])/total_files*100:.1f}%)")
        print(f"   验证集: {len(splits['val'])} 个文件 ({len(splits['val'])/total_files*100:.1f}%)")
        
        return splits
    
    def convert_labelme_to_coco_annotation(self, labelme_data, image_info):
        """
        将单个LabelMe标注转换为COCO格式
        
        Args:
            labelme_data (dict): LabelMe标注数据
            image_info (dict): 图片信息
            
        Returns:
            list: COCO格式的标注列表
        """
        annotations = []
        
        for shape in labelme_data.get('shapes', []):
            # 只处理矩形框
            if shape['shape_type'] != 'rectangle':
                continue
                
            # 获取类别ID
            label = shape['label']
            if label not in self.category_name_to_id:
                print(f"    警告: 未知类别 '{label}'，跳过该标注")
                continue
                
            category_id = self.category_name_to_id[label]
            
            # 获取矩形框坐标 (LabelMe格式: [[x1,y1], [x2,y2]])
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 确保坐标顺序正确
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            
            # 计算COCO格式所需的信息
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            # 创建COCO格式的标注
            annotation = {
                "id": self.annotation_id,
                "image_id": image_info['id'],
                "category_id": category_id,
                "bbox": [x_min, y_min, width, height],  # COCO格式: [x, y, width, height]
                "area": area,
                "iscrowd": 0,  # 目标检测通常设为0
                "segmentation": [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]  # 用矩形框的四个角点表示分割
            }
            
            annotations.append(annotation)
            self.annotation_id += 1
            
        return annotations
    
    def process_image(self, image_path, json_path):
        """
        处理单个图片和对应的标注文件
        
        Args:
            image_path (Path): 图片文件路径
            json_path (Path): LabelMe标注文件路径
            
        Returns:
            dict: COCO格式的图片信息
        """
        # 读取图片信息
        try:
            # 读取图片获取尺寸信息
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"   无法读取图片: {image_path}")
                return None
                
            height, width, _ = image.shape
            
            # 创建图片信息
            image_info = {
                "id": self.image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "date_captured": datetime.now().isoformat()
            }
            
            # 读取LabelMe标注文件
            with open(json_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            # 转换标注
            annotations = self.convert_labelme_to_coco_annotation(labelme_data, image_info)
            
            self.image_id += 1
            return image_info, annotations
            
        except Exception as e:
            print(f"   处理文件时出错 {image_path}: {str(e)}")
            return None
    
    def convert_spilt(self, file_pairs, split_name):
        print(f"开始转换 {split_name} 数据集...")

        self.image_id = 1  # 重置图片ID
        self.annotation_id = 1  # 重置注释ID

        coco_data = {
            "info": {
                "description": f"牙齿龋齿检测数据集 - {split_name}集",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Dental Detection Project",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "categories": self.categories,
            "images": [],
            "annotations": []
        }

        # 处理每个文件对
        successful_count = 0
        failed_count = 0
        
        for image_path, json_path in tqdm(file_pairs, desc=f"转换{split_name}集"):
            result = self.process_image(image_path, json_path)
            if result is not None:
                image_info, annotations = result
                coco_data["images"].append(image_info)
                coco_data["annotations"].extend(annotations)
                successful_count += 1
            else:
                failed_count += 1
        
        print(f" {split_name}集转换完成:")
        print(f"   成功: {successful_count} 个文件")
        print(f"   失败: {failed_count} 个文件")
        print(f"   图片数量: {len(coco_data['images'])}")
        print(f"   标注数量: {len(coco_data['annotations'])}")
        
        return coco_data
    
    def save_coco_json(self, coco_data, output_path):
        """
        保存COCO格式的JSON文件
        
        Args:
            coco_data (dict): COCO格式数据
            output_path (str): 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        
        print(f"   已保存: {output_path}")

    def save_splits_info(self, splits, output_dir):
        """
        保存数据集划分信息
        
        Args:
            splits (dict): 数据集划分信息
            output_dir (str): 输出目录
        """
        splits_info = {}
        for split_name, file_pairs in splits.items():
            splits_info[split_name] = [
                {
                    "image": str(img_path),
                    "annotation": str(json_path)
                }
                for img_path, json_path in file_pairs
            ]
        
        splits_file = Path(output_dir) / "splits_info.json"
        with open(splits_file, 'w', encoding='utf-8') as f:
            json.dump(splits_info, f, ensure_ascii=False, indent=2)
        
        print(f"已保存数据集划分信息: {splits_file}")

    def convert(self):
        print(f" 开始LabelMe到COCO格式转换...")
        print(f" 图片目录: {self.config['data']['images_dir']}")
        print(f" 标注目录: {self.config['data']['labelme_dir']}")
        print(f" 输出目录: {self.config['data']['annotations_dir']}")
        
        # 1. 收集文件
        file_pairs = self.collect_files(
            self.config['data']['images_dir'],
            self.config['data']['labelme_dir']
        )
        
        if not file_pairs:
            print("  没有找到匹配的图片-标注文件对！")
            return
        
        # 2. 划分数据集
        splits = self.split_dataset(
            file_pairs,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio']
        )
        
        # 3. 转换每个分割并保存
        annotations_dir = Path(self.config['data']['annotations_dir'])

        for split_name, file_pairs in splits.items():
            if file_pairs:  # 只处理非空的分割
                coco_data = self.convert_spilt(file_pairs, split_name)
                output_path = annotations_dir / f"{split_name}.json"
                self.save_coco_json(coco_data, output_path)
        
        # 4. 保存划分信息
        self.save_splits_info(splits, annotations_dir)
        
        # 5. 显示统计信息
        self.show_statistics(splits)
        
        print("\n  转换完成！")

    def show_statistics(self, splits):
        """显示转换统计信息"""
        print("\n 转换统计信息:")
        print("=" * 50)
        
        total_images = sum(len(file_pairs) for file_pairs in splits.values())
        print(f"总图片数量: {total_images}")
        
        for split_name, file_pairs in splits.items():
            if file_pairs:
                print(f"{split_name}集: {len(file_pairs)} 张图片")
        
        print(f"类别数量: {len(self.categories)}")
        print("类别列表:")
        for category in self.categories:
            print(f"  - {category['name']} (ID: {category['id']})")


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LabelMe到COCO格式转换工具')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='配置文件路径 (默认: config/config.yaml)'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        help='图片目录路径 (覆盖配置文件中的设置)'
    )
    parser.add_argument(
        '--labelme-dir',
        type=str,
        help='LabelMe标注目录路径 (覆盖配置文件中的设置)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='输出目录路径 (覆盖配置文件中的设置)'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print(f"  加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 使用命令行参数覆盖配置
        if args.images_dir:
            config['data']['images_dir'] = args.images_dir
        if args.labelme_dir:
            config['data']['labelme_dir'] = args.labelme_dir
        if args.output_dir:
            config['data']['annotations_dir'] = args.output_dir
        
        # 检查必要的路径
        required_paths = ['images_dir', 'labelme_dir']
        for path_key in required_paths:
            path = config['data'][path_key]
            if not os.path.exists(path):
                print(f"   路径不存在: {path}")
                return
        
        # 设置随机种子以确保可重现的数据集划分
        random.seed(config['data'].get('random_seed', 42))
        
        # 创建转换器并执行转换
        converter = LabelMeToCOCOConverter(config)
        converter.convert()
        
    except FileNotFoundError as e:
        print(f"   文件未找到: {e}")
    except yaml.YAMLError as e:
        print(f"   配置文件格式错误: {e}")
    except Exception as e:
        print(f"   转换过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()