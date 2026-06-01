import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import torch

class NailDetector:
    def __init__(self, config, model_path: Optional[str] = None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(config.YOLO_MODEL_NAME)
        
        self.model.to(self.device)
        
    def train_yolo(self, dataset_config: str):
        """Train YOLOv8 model for nail detection"""
        print("Training YOLOv8 model for nail detection...")
        
        results = self.model.train(
            data=dataset_config,
            epochs=self.config.YOLO_EPOCHS,
            imgsz=self.config.YOLO_IMAGE_SIZE,
            batch=16,
            save=True,
            project=self.config.YOLO_DIR,
            name='nail_detection',
            device=self.device,
            patience=10,
            optimizer='AdamW'
        )
        
        # Save the best model
        best_model_path = os.path.join(self.config.YOLO_DIR, 'nail_detection', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            os.rename(best_model_path, self.config.YOLO_CUSTOM_MODEL_PATH)
        
        return results
    
    def detect_nails(self, image_path: str, save: bool = False) -> Tuple[np.ndarray, List]:
        """Detect nails in an image and return cropped nail regions"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        
        # Perform detection
        results = self.model(image, conf=self.config.YOLO_CONFIDENCE)
        
        cropped_nails = []
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0])
                    
                    # Crop nail region with padding
                    padding = 10
                    h, w = image.shape[:2]
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(w, x2 + padding)
                    y2_padded = min(h, y2 + padding)
                    
                    cropped_nail = original_image[y1_padded:y2_padded, x1_padded:x2_padded]
                    
                    if cropped_nail.size > 0:
                        cropped_nails.append(cropped_nail)
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'cropped_image': cropped_nail
                        })
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f'Nail: {confidence:.2f}', 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save:
            output_path = os.path.join(self.config.YOLO_DIR, 'detection_results', 
                                     os.path.basename(image_path))
            cv2.imwrite(output_path, image)
        
        return image, detections
    
    def extract_nails_from_directory(self, input_dir: str, output_dir: str):
        """Extract and save nail regions from all images in directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(supported_formats)]
        
        total_detections = 0
        
        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            
            try:
                _, detections = self.detect_nails(image_path)
                
                for i, detection in enumerate(detections):
                    # Save cropped nail image
                    filename = f"{os.path.splitext(image_file)[0]}_nail_{i}.jpg"
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, detection['cropped_image'])
                    total_detections += 1
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"Extracted {total_detections} nail images to {output_dir}")
        return total_detections