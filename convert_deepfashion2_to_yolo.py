# convert_deepfashion2_to_yolo.py

import os
import json
import cv2
from pathlib import Path
import yaml
from tqdm import tqdm
import shutil

class DeepFashion2ToYOLO:
    def __init__(self, data_dir="./data", output_dir="./yolo_dataset"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.categories = {
            1: 'short_sleeved_shirt',
            2: 'long_sleeved_shirt', 
            3: 'short_sleeved_outwear',
            4: 'long_sleeved_outwear',
            5: 'vest',
            6: 'sling',
            7: 'shorts',
            8: 'trousers',
            9: 'skirt',
            10: 'short_sleeved_dress',
            11: 'long_sleeved_dress',
            12: 'vest_dress',
            13: 'sling_dress'
        }
        self.setup_directories()
    
    def setup_directories(self):
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def polygon_to_yolo_segmentation(self, polygon, img_width, img_height):
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            return []
        normalized_seg = []
        for i in range(0, len(polygon), 2):
            # Clamp coordinates to [0, 1] range
            x = max(0.0, min(1.0, polygon[i] / img_width))
            y = max(0.0, min(1.0, polygon[i + 1] / img_height))
            normalized_seg.extend([x, y])
        return normalized_seg


    def convert_annotation(self, anno_path, img_width, img_height):
        try:
            with open(anno_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {anno_path}: {e}")
            return []

        yolo_annotations = []

        # Loop over all keys, looking for item1, item2, ...
        for key in data:
            if not key.startswith('item'):
                continue
            item = data[key]
            category_id = item.get('category_id')
            if category_id not in self.categories:
                continue
            class_id = category_id - 1

            segmentation = item.get('segmentation', [])
            polygons = []
            if isinstance(segmentation, list) and len(segmentation) > 0:
                if isinstance(segmentation[0], list):
                    polygons = segmentation
                elif isinstance(segmentation[0], (int, float)):
                    polygons = [segmentation]
            # For each polygon, create a YOLO annotation
            for polygon in polygons:
                yolo_seg = self.polygon_to_yolo_segmentation(polygon, img_width, img_height)
                if len(yolo_seg) < 6:
                    continue
                annotation_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in yolo_seg])
                yolo_annotations.append(annotation_line)
        return yolo_annotations

    def convert_dataset(self, train_split=0.8, sample_ratio=1.0):
        anno_dir = self.data_dir / 'train' / 'annos'
        image_dir = self.data_dir / 'train' / 'image'
        print(f"Annotation directory: {anno_dir}")
        print(f"Image directory: {image_dir}")

        anno_files = list(anno_dir.glob('*.json'))
        if sample_ratio < 1.0:
            sample_size = int(len(anno_files) * sample_ratio)
            anno_files = anno_files[:sample_size]
        print(f"Converting {len(anno_files)} images...")

        split_idx = int(len(anno_files) * train_split)
        train_files = anno_files[:split_idx]
        val_files = anno_files[split_idx:]

        train_converted = self._convert_split(train_files, image_dir, 'train')
        val_converted = self._convert_split(val_files, image_dir, 'val')
        self.create_yaml_config()
        print(f"Conversion complete!")
        print(f"Train images converted: {train_converted}/{len(train_files)}")
        print(f"Val images converted: {val_converted}/{len(val_files)}")
        print(f"Dataset saved to: {self.output_dir}")
        self.verify_conversion()

    def _convert_split(self, anno_files, image_dir, split):
        converted_count = 0
        for anno_file in tqdm(anno_files, desc=f"Converting {split}"):
            image_id = anno_file.stem
            image_path = image_dir / f"{image_id}.jpg"
            if not image_path.exists():
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            yolo_annotations = self.convert_annotation(anno_file, img_width, img_height)
            if not yolo_annotations:
                continue
            dst_image_path = self.output_dir / split / 'images' / f"{image_id}.jpg"
            shutil.copy2(image_path, dst_image_path)
            dst_label_path = self.output_dir / split / 'labels' / f"{image_id}.txt"
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            converted_count += 1
        print(f"Converted {converted_count} {split} images")
        return converted_count

    def verify_conversion(self):
        train_images = len(list((self.output_dir / 'train' / 'images').glob('*.jpg')))
        train_labels = len(list((self.output_dir / 'train' / 'labels').glob('*.txt')))
        val_images = len(list((self.output_dir / 'val' / 'images').glob('*.jpg')))
        val_labels = len(list((self.output_dir / 'val' / 'labels').glob('*.txt')))
        print(f"\nVerification:")
        print(f"Train: {train_images} images, {train_labels} labels")
        print(f"Val: {val_images} images, {val_labels} labels")
        if train_images == 0 and val_images == 0:
            print("ERROR: No images were converted! Check your data format.")
        elif train_images != train_labels or val_images != val_labels:
            print("WARNING: Mismatch between images and labels count.")
        else:
            print("SUCCESS: Conversion completed successfully!")

    def create_yaml_config(self):
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 13,
            'names': list(self.categories.values())
        }
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created config file: {yaml_path}")

if __name__ == "__main__":
    converter = DeepFashion2ToYOLO(
        data_dir="./data",
        output_dir="./yolo_dataset"
    )
    converter.convert_dataset(train_split=0.8, sample_ratio=1.0)
