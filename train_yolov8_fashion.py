# train_yolov8_fashion.py
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml
import time
from datetime import datetime
import argparse

class YOLOv8FashionTrainer:
    def __init__(self, dataset_path="./yolo_dataset", model_size="n"):
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None
        
        # Check if dataset exists
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path {dataset_path} does not exist. Run convert_deepfashion2_to_yolo.py first.")
        
        # Load dataset config
        self.config_path = self.dataset_path / 'data.yaml'
        if not self.config_path.exists():
            raise ValueError(f"Dataset config {self.config_path} not found.")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Dataset: {self.config['nc']} classes")
        print(f"Classes: {self.config['names']}")
    
    def load_model(self, pretrained=True):
        """Load YOLOv8 segmentation model"""
        if pretrained:
            model_name = f'yolov8{self.model_size}-seg.pt'
            print(f"Loading pretrained model: {model_name}")
        else:
            model_name = f'yolov8{self.model_size}-seg.yaml'
            print(f"Creating new model: {model_name}")
        
        self.model = YOLO(model_name)
        return self.model
    
    def train(self, epochs=50, batch_size=16, img_size=640, device='auto', 
              patience=50, save_period=10, workers=8, cache=None, quick_test=False):
        """Train YOLOv8 model with full optimization support"""
        
        if self.model is None:
            self.load_model(pretrained=True)
        
        # Adjust parameters for quick test
        if quick_test:
            epochs = 10
            batch_size = 8
            img_size = 416
            patience = 5
            print("Quick test mode: reduced epochs and image size")
        
        # Check device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Training on device: {device}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
        print(f"Workers: {workers}, Cache: {cache}")
        
        # Start training
        start_time = time.time()
        
        results = self.model.train(
            data=str(self.config_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            patience=patience,
            save_period=save_period,
            workers=workers,
            cache=cache if cache else False,
            project='runs/segment',
            name='fashion_training',
            exist_ok=True,
            pretrained=True,
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,
            save_json=False,
            save_hybrid=False,
            conf=None,
            iou=0.7,
            max_det=300,
            half=False,
            dnn=False,
            augment=True,
            agnostic_nms=False,
            retina_masks=False,
            format='torchscript',
            keras=False,
            optimize=False,
            int8=False,
            dynamic=False,
            simplify=False,
            opset=None,
            workspace=4,
            nms=False,
            # Additional optimizations
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,  # Use full dataset
            profile=False,  # Disable profiling for speed
            freeze=None,  # Don't freeze layers
            multi_scale=False,  # Disable multi-scale for speed
            copy_paste=0.0,  # Disable copy-paste augmentation
            auto_augment='randaugment',
            erasing=0.4,
            crop_fraction=1.0,
            # Data loading optimizations
            rect=False,  # Disable rectangular training for speed
            cos_lr=False,  # Use step LR instead of cosine
            close_mosaic=10,  # Close mosaic augmentation in last 10 epochs
            resume=False,
            # Validation settings
            split='val',
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            visualize=False,
            # Speed optimizations
            deterministic=True,
            single_cls=False,
            verbose=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/3600:.2f} hours")
        
        return results
    
    def validate(self, model_path=None):
        """Validate the trained model"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        if model is None:
            raise ValueError("No model loaded. Train first or provide model_path.")
        
        results = model.val(
            data=str(self.config_path),
            split='val',
            save_json=True,
            save_hybrid=False,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=False,
            device='auto',
            dnn=False,
            plots=True,
            rect=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            visualize=False,
            augment=False,
            agnostic_nms=False,
            retina_masks=False,
            format='torchscript',
            keras=False,
            optimize=False,
            int8=False,
            dynamic=False,
            simplify=False,
            opset=None,
            workspace=4,
            nms=False
        )
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on DeepFashion2 dataset')
    parser.add_argument('--dataset', default='./yolo_dataset', help='Path to YOLO dataset')
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'], 
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--cache', type=str, default=None, choices=[None, 'ram', 'disk'], 
                       help='Cache images in RAM or disk for faster training')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with reduced parameters')
    parser.add_argument('--validate-only', help='Path to model for validation only')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = YOLOv8FashionTrainer(
        dataset_path=args.dataset,
        model_size=args.model_size
    )
    
    if args.validate_only:
        # Validation only
        print("Running validation...")
        results = trainer.validate(args.validate_only)
        print("Validation completed")
    else:
        # Training
        print("Starting training...")
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            patience=args.patience,
            save_period=args.save_period,
            workers=args.workers,
            cache=args.cache,
            quick_test=args.quick_test
        )
        
        # Validate after training
        print("Running validation...")
        val_results = trainer.validate()
        print("Training and validation completed")

if __name__ == "__main__":
    main()
