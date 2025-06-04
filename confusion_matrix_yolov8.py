# confusion_matrix_updated.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import cv2
from tqdm import tqdm

class UpdatedYOLOConfusionMatrix:
    def __init__(self):
        self.class_names = [
            'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
            'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers',
            'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
        ]
        
    def extract_predictions_from_inference(self, inference_dir="./yolo_dataset/val/inference_results"):
        """Extract predictions from your validation inference results"""
        
        inference_dir = Path(inference_dir)
        predictions = []
        
        # Look for batch_results.json in the validation inference directory
        batch_file = inference_dir / "batch_results.json"
        
        if batch_file.exists():
            print(f"Loading predictions from {batch_file}")
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                
                for image_name, detections in tqdm(batch_data.items(), desc="Processing predictions"):
                    image_id = image_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                    
                    if isinstance(detections, list):
                        for detection in detections:
                            if isinstance(detection, dict) and 'class_id' in detection:
                                predictions.append({
                                    'image_name': image_id,
                                    'predicted_class_id': int(detection['class_id']),
                                    'predicted_class_name': detection.get('class_name', f"class_{detection['class_id']}"),
                                    'confidence': float(detection.get('confidence', 0.0)),
                                    'bbox': detection.get('bbox', [])
                                })
                
                print(f"Extracted {len(predictions)} predictions from validation inference")
                return pd.DataFrame(predictions)
                
            except Exception as e:
                print(f"Error processing {batch_file}: {e}")
                return pd.DataFrame()
        else:
            print(f"No batch_results.json found in {inference_dir}")
            print(f"Available files: {list(inference_dir.glob('*'))}")
            return pd.DataFrame()
    
    def extract_ground_truth_from_yolo(self, yolo_dataset_dir="./yolo_dataset"):
        """Extract ground truth from YOLO validation labels"""
        
        val_images_dir = Path(yolo_dataset_dir) / 'val' / 'images'
        val_labels_dir = Path(yolo_dataset_dir) / 'val' / 'labels'
        
        if not val_images_dir.exists() or not val_labels_dir.exists():
            print(f"YOLO validation directories not found:")
            print(f"  Images: {val_images_dir}")
            print(f"  Labels: {val_labels_dir}")
            return pd.DataFrame()
        
        ground_truth = []
        
        # Get all label files
        label_files = list(val_labels_dir.glob("*.txt"))
        print(f"Processing {len(label_files)} YOLO label files...")
        
        for label_file in tqdm(label_files, desc="Processing labels"):
            image_name = label_file.stem
            image_path = val_images_dir / f"{image_name}.jpg"
            
            # Check if corresponding image exists
            if not image_path.exists():
                # Try other extensions
                alt_paths = [
                    val_images_dir / f"{image_name}.jpeg",
                    val_images_dir / f"{image_name}.png"
                ]
                image_path = next((p for p in alt_paths if p.exists()), None)
                if not image_path:
                    continue
            
            # Read image dimensions for bbox conversion
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]
            except:
                continue
            
            # Read YOLO label file
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id + 4 bbox coordinates minimum
                        class_id = int(parts[0])
                        
                        if 0 <= class_id < len(self.class_names):
                            # Convert YOLO format to absolute coordinates
                            center_x = float(parts[1]) * img_width
                            center_y = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            x_min = center_x - width / 2
                            y_min = center_y - height / 2
                            x_max = center_x + width / 2
                            y_max = center_y + height / 2
                            
                            bbox = [x_min, y_min, x_max, y_max]
                            
                            ground_truth.append({
                                'image_name': image_name,
                                'true_class_id': class_id,
                                'true_class_name': self.class_names[class_id],
                                'bbox': bbox
                            })
                        
            except Exception as e:
                continue
        
        print(f"Extracted {len(ground_truth)} ground truth annotations")
        return pd.DataFrame(ground_truth)
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        if len(box1) != 4 or len(box2) != 4:
            return 0.0
        
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_predictions_to_ground_truth(self, predictions_df, ground_truth_df, iou_threshold=0.3):
        """Match predictions to ground truth with multiple strategies"""
        
        matched_pairs = []
        
        # Get common images
        pred_images = set(predictions_df['image_name'])
        gt_images = set(ground_truth_df['image_name'])
        common_images = pred_images.intersection(gt_images)
        
        print(f"Prediction images: {len(pred_images)}")
        print(f"Ground truth images: {len(gt_images)}")
        print(f"Common images: {len(common_images)}")
        
        if len(common_images) == 0:
            print("No common images found! Checking image name formats...")
            print("Sample prediction images:", list(pred_images)[:5])
            print("Sample ground truth images:", list(gt_images)[:5])
            return pd.DataFrame()
        
        # Strategy 1: IoU-based matching
        iou_matches = 0
        class_matches = 0
        
        for image_name in tqdm(common_images, desc="Matching predictions"):
            pred_subset = predictions_df[predictions_df['image_name'] == image_name]
            gt_subset = ground_truth_df[ground_truth_df['image_name'] == image_name]
            
            if len(gt_subset) == 0:
                continue
            
            # Try IoU matching first
            for _, pred in pred_subset.iterrows():
                best_iou = 0
                best_match = None
                
                for _, gt in gt_subset.iterrows():
                    if len(pred['bbox']) == 4 and len(gt['bbox']) == 4:
                        iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_match = gt
                
                if best_match is not None:
                    matched_pairs.append({
                        'image_name': image_name,
                        'predicted_class_id': pred['predicted_class_id'],
                        'predicted_class_name': pred['predicted_class_name'],
                        'true_class_id': best_match['true_class_id'],
                        'true_class_name': best_match['true_class_name'],
                        'confidence': pred['confidence'],
                        'iou': best_iou,
                        'match_type': 'iou'
                    })
                    iou_matches += 1
                else:
                    # Strategy 2: Class-based matching for same image
                    class_match = gt_subset[gt_subset['true_class_id'] == pred['predicted_class_id']]
                    if len(class_match) > 0:
                        best_class_match = class_match.iloc[0]
                        matched_pairs.append({
                            'image_name': image_name,
                            'predicted_class_id': pred['predicted_class_id'],
                            'predicted_class_name': pred['predicted_class_name'],
                            'true_class_id': best_class_match['true_class_id'],
                            'true_class_name': best_class_match['true_class_name'],
                            'confidence': pred['confidence'],
                            'iou': 0.0,
                            'match_type': 'class'
                        })
                        class_matches += 1
        
        print(f"IoU-based matches: {iou_matches}")
        print(f"Class-based matches: {class_matches}")
        print(f"Total matches: {len(matched_pairs)}")
        
        return pd.DataFrame(matched_pairs)
    
    def create_confusion_matrix(self, matched_df, save_plots=True):
        """Create confusion matrix with detailed analysis"""
        
        if len(matched_df) == 0:
            print("No matched predictions found. Cannot create confusion matrix.")
            return None, 0, []
        
        # Extract true and predicted labels
        y_true = matched_df['true_class_id'].values
        y_pred = matched_df['predicted_class_id'].values
        
        # Get unique classes present in the data
        unique_classes = sorted(list(set(y_true) | set(y_pred)))
        class_labels = [self.class_names[i] for i in unique_classes]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        # Calculate accuracy
        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        
        # Print detailed statistics
        print(f"\nConfusion Matrix Analysis:")
        print(f"Total predictions matched: {len(matched_df)}")
        print(f"Unique classes in predictions: {len(set(y_pred))}")
        print(f"Unique classes in ground truth: {len(set(y_true))}")
        print(f"Overall accuracy: {accuracy:.3f}")
        
        # Per-class analysis
        print(f"\nPer-class breakdown:")
        for i, class_idx in enumerate(unique_classes):
            class_name = self.class_names[class_idx]
            true_count = np.sum(y_true == class_idx)
            pred_count = np.sum(y_pred == class_idx)
            correct = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
            
            if true_count > 0:
                recall = correct / true_count
                print(f"  {class_name}: {correct}/{true_count} correct (recall: {recall:.3f})")
        
        if save_plots:
            # Plot confusion matrix
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f'YOLOv8 Fashion Classification Confusion Matrix\nOverall Accuracy: {accuracy:.3f}')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('validation_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))
        
        # Calculate per-class accuracy
        per_class_accuracy = []
        for i, class_idx in enumerate(unique_classes):
            if i < cm.shape[0] and cm.sum(axis=1)[i] > 0:
                per_class_accuracy.append(cm[i, i] / cm.sum(axis=1)[i])
            else:
                per_class_accuracy.append(0)
        
        return cm, accuracy, per_class_accuracy
    
    def run_analysis(self):
        """Run complete analysis with your validation inference results"""
        
        print("🔍 Step 1: Extracting predictions from validation inference...")
        predictions_df = self.extract_predictions_from_inference("./yolo_dataset/val/inference_results")
        print(f"   Found {len(predictions_df)} predictions")
        
        if len(predictions_df) == 0:
            print("❌ No predictions found. Check your inference results directory.")
            return None, 0, []
        
        print("\n📋 Step 2: Extracting ground truth from YOLO validation...")
        ground_truth_df = self.extract_ground_truth_from_yolo("./yolo_dataset")
        print(f"   Found {len(ground_truth_df)} ground truth annotations")
        
        if len(ground_truth_df) == 0:
            print("❌ No ground truth found.")
            return None, 0, []
        
        print("\n🔗 Step 3: Matching predictions to ground truth...")
        matched_df = self.match_predictions_to_ground_truth(predictions_df, ground_truth_df, iou_threshold=0.3)
        print(f"   Matched {len(matched_df)} prediction-ground truth pairs")
        
        if len(matched_df) == 0:
            print("❌ No matches found. Try lowering IoU threshold.")
            return None, 0, []
        
        print("\n📊 Step 4: Generating confusion matrix...")
        cm, accuracy, per_class_acc = self.create_confusion_matrix(matched_df)
        
        print("\n💾 Step 5: Saving results...")
        matched_df.to_csv('validation_confusion_matrix_matches.csv', index=False)
        
        return cm, accuracy, per_class_acc

def main():
    print("🚀 Updated YOLOv8 Validation Confusion Matrix Generator")
    print("=" * 60)
    
    analyzer = UpdatedYOLOConfusionMatrix()
    
    # Run analysis on your validation inference results
    cm, accuracy, per_class_acc = analyzer.run_analysis()
    
    if cm is not None and len(per_class_acc) > 0:
        print(f"\n🎉 Analysis Complete!")
        print(f"📈 Overall Accuracy: {accuracy:.1%}")
        if len(per_class_acc) > 0:
            print(f"🏆 Best performing class: {analyzer.class_names[np.argmax(per_class_acc)]}")
            print(f"📉 Worst performing class: {analyzer.class_names[np.argmin(per_class_acc)]}")
    else:
        print("❌ Analysis failed.")

if __name__ == "__main__":
    main()
