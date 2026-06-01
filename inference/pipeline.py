import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AnaemiaPipeline:
    """
    End-to-end pipeline:
    Raw hand image → YOLOv8 nail detection → crop → EfficientNet classification
    """
    def __init__(self, config, cnn_model_path, yolo_model_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Pipeline running on: {self.device}")

        # ── Load YOLOv8 nail detector ──────────────────────────────────────
        print(f"Loading YOLO from: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)

        # ── Load EfficientNet anaemia classifier ───────────────────────────
        print(f"Loading CNN from: {cnn_model_path}")
        from models.anaemia_model import EfficientNetAnemia
        self.cnn = EfficientNetAnemia(num_classes=config.NUM_CLASSES)
        self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn.to(self.device)
        self.cnn.eval()

        # ── Preprocessing for CNN ──────────────────────────────────────────
        self.transform = A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def detect_nails(self, image_bgr):
        """Run YOLOv8 on image, return list of cropped nail regions + boxes"""
        results = self.yolo(image_bgr, conf=0.3, verbose=False)
        crops = []
        boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Add padding around nail
                pad = 10
                h, w = image_bgr.shape[:2]
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)

                crop = image_bgr[y1p:y2p, x1p:x2p]
                if crop.size > 0:
                    crops.append(crop)
                    boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })

        return crops, boxes

    def classify_nail(self, nail_crop_bgr):
        """Run EfficientNet on a single nail crop"""
        # BGR → RGB
        nail_rgb = cv2.cvtColor(nail_crop_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        transformed = self.transform(image=nail_rgb)
        tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.cnn(tensor)
            probs   = torch.softmax(outputs, dim=1)
            pred    = torch.argmax(probs, dim=1).item()

        return {
            'class':         self.config.CLASS_NAMES[pred],
            'confidence':    float(probs[0][pred]),
            'probabilities': {
                self.config.CLASS_NAMES[i]: float(p)
                for i, p in enumerate(probs[0])
            }
        }

    def predict(self, image_path):
        """
        Full pipeline on a single image.
        Returns prediction dict with detection + classification results.
        """
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {'error': f'Could not load image: {image_path}'}

        original = image_bgr.copy()

        # ── Step 1: Detect nails ───────────────────────────────────────────
        crops, boxes = self.detect_nails(image_bgr)

        if len(crops) == 0:
            # No nail detected — classify full image as fallback
            print("⚠️  No nail detected, classifying full image as fallback")
            crops  = [image_bgr]
            boxes  = [{'bbox': [0, 0, image_bgr.shape[1], image_bgr.shape[0]],
                       'confidence': 0.0}]
            fallback = True
        else:
            fallback = False
            print(f"✅ Detected {len(crops)} nail region(s)")

        # ── Step 2: Classify each nail crop ───────────────────────────────
        nail_results = []
        for i, (crop, box) in enumerate(zip(crops, boxes)):
            clf = self.classify_nail(crop)
            nail_results.append({
                'nail_id':          i + 1,
                'bbox':             box['bbox'],
                'yolo_confidence':  box['confidence'],
                'prediction':       clf['class'],
                'cnn_confidence':   clf['confidence'],
                'probabilities':    clf['probabilities'],
            })

        # ── Step 3: Aggregate — majority vote across nails ─────────────────
        anaemic_votes     = sum(1 for r in nail_results if r['prediction'] == 'anaemic')
        non_anaemic_votes = len(nail_results) - anaemic_votes
        final_class       = 'anaemic' if anaemic_votes >= non_anaemic_votes else 'non_anaemic'
        avg_confidence    = np.mean([r['cnn_confidence'] for r in nail_results])

        # ── Step 4: Draw annotated image ──────────────────────────────────
        annotated = self.draw_results(original, nail_results, final_class)

        return {
            'image_path':      image_path,
            'final_prediction': final_class,
            'confidence':       float(avg_confidence),
            'nail_count':       len(nail_results),
            'nail_results':     nail_results,
            'fallback_used':    fallback,
            'annotated_image':  annotated,
        }

    def draw_results(self, image, nail_results, final_class):
        """Draw bounding boxes and predictions on image"""
        annotated = image.copy()

        for r in nail_results:
            x1, y1, x2, y2 = r['bbox']
            color = (0, 0, 255) if r['prediction'] == 'anaemic' else (0, 255, 0)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{r['prediction']} {r['cnn_confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(annotated, label,
                       (x1+2, y1-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw final prediction banner at top
        banner_color = (0, 0, 200) if final_class == 'anaemic' else (0, 180, 0)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), banner_color, -1)
        cv2.putText(annotated,
                   f"FINAL: {final_class.upper()}",
                   (10, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return annotated

    def predict_batch(self, image_dir, output_dir=None):
        """Run pipeline on all images in a directory"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        supported = ('.jpg', '.jpeg', '.png')
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(supported)
        ]

        print(f"\n🔍 Running pipeline on {len(image_files)} images...")
        results = []

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            result   = self.predict(img_path)

            if 'error' not in result:
                results.append(result)
                print(f"  {img_file:50s} → {result['final_prediction']:12s} "
                      f"({result['confidence']:.1%} conf, "
                      f"{result['nail_count']} nail(s) detected)")

                # Save annotated image
                if output_dir and result['annotated_image'] is not None:
                    out_path = os.path.join(output_dir, f"annotated_{img_file}")
                    cv2.imwrite(out_path, result['annotated_image'])

        # Summary
        if results:
            anaemic_count = sum(1 for r in results if r['final_prediction'] == 'anaemic')
            print(f"\n📊 Batch Summary:")
            print(f"   Total images  : {len(results)}")
            print(f"   Anaemic       : {anaemic_count}")
            print(f"   Non-anaemic   : {len(results) - anaemic_count}")
            print(f"   Avg confidence: {np.mean([r['confidence'] for r in results]):.1%}")
            if output_dir:
                print(f"   Annotated images saved to: {output_dir}")

        return results