# inference_yolov8_fashion.py
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import argparse
from color_extractor import AdvancedColorExtractor

class YOLOv8FashionDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize the fashion detector with color extraction"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.color_extractor = AdvancedColorExtractor(n_colors=5)
        
        # DeepFashion2 class names
        self.class_names = [
            'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
            'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers',
            'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
        ]
    
    def detect_and_extract_colors(self, image_path, save_results=True):
        """Detect clothing items and extract their colors"""
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(image_path, conf=self.confidence_threshold)
        
        detections = []
        
        for result in results:
            if result.masks is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                masks = result.masks.data.cpu().numpy()
                
                for i, (box, conf, class_id, mask) in enumerate(zip(boxes, confidences, class_ids, masks)):
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Resize mask to match image dimensions
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    
                    # Extract colors from the masked region
                    colors, percentages, color_names = self.color_extractor.extract_colors_from_mask(
                        image_rgb, mask_resized
                    )
                    
                    # Get color information
                    color_info = self.color_extractor.get_color_info(colors, percentages, color_names)
                    
                    detection = {
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                        'confidence': float(conf),
                        'bbox': [float(x) for x in box],
                        'colors': color_info,
                        'dominant_color': color_info[0] if color_info else None
                    }
                    
                    detections.append(detection)
        
        # Save results if requested
        if save_results:
            self.save_results(image_path, detections, image_rgb, results)
        
        return detections
    
    def save_results(self, image_path, detections, image_rgb, yolo_results):
        """Save detection results and annotated image"""
        
        image_path = Path(image_path)
        output_dir = Path("inference_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / f"{image_path.stem}_results.json"
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=2)
        
        # Save annotated image
        annotated_image = self.draw_annotations(image_rgb, detections)
        annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Save YOLO visualization
        yolo_vis_path = output_dir / f"{image_path.stem}_yolo.jpg"
        if yolo_results:
            yolo_annotated = yolo_results[0].plot()
            cv2.imwrite(str(yolo_vis_path), yolo_annotated)
        
        print(f"Results saved to {output_dir}")
        print(f"JSON: {json_path}")
        print(f"Annotated: {annotated_path}")
        print(f"YOLO visualization: {yolo_vis_path}")
    
    def draw_annotations(self, image, detections):
        """Draw bounding boxes and color information on image"""
        
        annotated = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            colors = detection['colors']
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw class label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw color palette
            if colors:
                palette_height = 20
                palette_y = y2 + 5
                total_width = x2 - x1
                
                current_x = x1
                for color_info in colors[:3]:  # Show top 3 colors
                    color_rgb = color_info['rgb']
                    percentage = color_info['percentage']
                    color_width = int(total_width * percentage)
                    
                    cv2.rectangle(annotated, 
                                 (current_x, palette_y), 
                                 (current_x + color_width, palette_y + palette_height),
                                 color_rgb, -1)
                    
                    current_x += color_width
                
                # Add color text
                if colors:
                    color_text = f"Colors: {', '.join([c['name'] for c in colors[:2]])}"
                    cv2.putText(annotated, color_text, (x1, palette_y + palette_height + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return annotated
    
    def batch_process(self, image_dir, output_dir=None):
        """Process multiple images in a directory"""
        
        image_dir = Path(image_dir)
        if output_dir is None:
            output_dir = image_dir / "inference_results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        all_results = {}
        
        for image_path in image_files:
            print(f"Processing: {image_path.name}")
            try:
                detections = self.detect_and_extract_colors(image_path, save_results=False)
                all_results[image_path.name] = detections
                
                # Save individual results
                self.save_results(image_path, detections, 
                                cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB),
                                self.model(str(image_path)))
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        # Save batch summary
        summary_path = output_dir / "batch_results.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Batch processing complete. Results saved to {output_dir}")
        return all_results

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Fashion Detection with Color Extraction')
    parser.add_argument('--model', required=True, help='Path to trained YOLOv8 model')
    parser.add_argument('--image', help='Path to single image')
    parser.add_argument('--image-dir', help='Path to directory of images')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create detector
    detector = YOLOv8FashionDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    if args.image:
        # Process single image
        print(f"Processing image: {args.image}")
        detections = detector.detect_and_extract_colors(args.image)
        
        print(f"\nDetected {len(detections)} clothing items:")
        for i, detection in enumerate(detections):
            print(f"\n{i+1}. {detection['class_name']} (confidence: {detection['confidence']:.2f})")
            print("   Colors:")
            for color in detection['colors']:
                print(f"     - {color['name']}: {color['percentage']:.1%} ({color['hex']})")
    
    elif args.image_dir:
        # Process directory of images
        print(f"Processing directory: {args.image_dir}")
        results = detector.batch_process(args.image_dir, args.output_dir)
        print(f"Processed {len(results)} images")
    
    else:
        print("Please provide either --image or --image-dir")

if __name__ == "__main__":
    main()
