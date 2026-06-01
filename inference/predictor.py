import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

class AnemiaPredictor:
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.anaemia_model import SimpleCNN
        self.model = SimpleCNN(num_classes=config.NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def predict(self, image_path):
        """Predict anemia from a single image"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image from {image_path}"}
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # Apply transformations
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
            
            confidence = probabilities[0][prediction.item()].item()
            class_name = self.config.CLASS_NAMES[prediction.item()]
            
            return {
                'class': class_name,
                'confidence': float(confidence),
                'probabilities': {
                    self.config.CLASS_NAMES[i]: float(prob) 
                    for i, prob in enumerate(probabilities.cpu().numpy()[0])
                },
                'original_size': original_size
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}