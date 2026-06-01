import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from training.trainer import AnemiaTrainer
from inference.predictor import AnemiaPredictor
from utils.data_loader import DataManager
from yolov8.nail_detector import NailDetector

# import OpenVINO to avoid errors during training
try:
    from openvino.model_optimizer import OpenVINOPredictor
    OPENVINO_AVAILABLE = True
    print("✅ OpenVINO loaded successfully")
except ImportError as e:
    print(f"⚠️  OpenVINO not available: {e}")
    OPENVINO_AVAILABLE = False
except AttributeError as e:
    print(f"⚠️  OpenVINO compatibility issue: {e}")
    OPENVINO_AVAILABLE = False

def main():
    try:
        parser = argparse.ArgumentParser(description='Anemia Detection from Nail Images')
        parser.add_argument('--mode', type=str, required=True, 
                           choices=['train', 'predict', 'evaluate', 'train_yolo', 'extract_nails'],
                           help='Mode: train, predict, evaluate, train_yolo, or extract_nails')
        parser.add_argument('--data_path', type=str, 
                           default='data/', help='Path to dataset')
        parser.add_argument('--model_path', type=str, 
                           default='models/anemia_model.pth', help='Model path')
        parser.add_argument('--image_path', type=str, 
                           help='Single image path for prediction')
        parser.add_argument('--model_type', type=str, 
                           default='simple', choices=['simple', 'efficientnet', 'resnet'],
                           help='Type of model to use for training')
        parser.add_argument('--use_nail_detection', action='store_true',
                           help='Use YOLO nail detection during training/prediction')
        parser.add_argument('--use_openvino', action='store_true',
                           help='Use OpenVINO optimized model for inference')
        
        args = parser.parse_args()
        
        config = Config()
        
        if args.mode == 'train':
            print(f"Starting training process with {args.model_type} model...")
            trainer = AnemiaTrainer(config, model_type=args.model_type)
            trainer.train(use_nail_detection=args.use_nail_detection)
            
        elif args.mode == 'train_yolo':
            print("Training YOLOv8 model for nail detection...")
            detector = NailDetector(config)
            # need to provide a dataset config file for YOLO training
            dataset_config = "yolov8/data_config.yaml"  # You need to create this
            detector.train_yolo(dataset_config)
            
        elif args.mode == 'extract_nails':
            print("Extracting nails from raw images...")
            detector = NailDetector(config)
            raw_data_dir = "data/raw"  # Directory containing raw images
            detector.extract_nails_from_directory(raw_data_dir, config.DATA_DIR)
            
        elif args.mode == 'predict':
          if not args.image_path:
            raise ValueError("Please provide --image_path for prediction")

          from inference.pipeline import AnaemiaPipeline

          yolo_path = os.path.join(config.YOLO_DIR, 'nail_detector', 'weights', 'best.pt')
          cnn_path  = config.MODEL_SAVE_PATH.replace('.pth', f'_{args.model_type}.pth')

          pipeline = AnaemiaPipeline(config, cnn_path, yolo_path)
          result   = pipeline.predict(args.image_path)

          print(f"\n{'='*40}")
          print(f"PREDICTION RESULT")
          print(f"{'='*40}")
          print(f"Image      : {args.image_path}")
          print(f"Prediction : {result['final_prediction'].upper()}")
          print(f"Confidence : {result['confidence']:.1%}")
          print(f"Nails found: {result['nail_count']}")
          for r in result['nail_results']:
              print(f"  Nail {r['nail_id']}: {r['prediction']} ({r['cnn_confidence']:.1%})")
          print(f"{'='*40}")
            
        elif args.mode == 'evaluate':
            print("Evaluation mode")
            # implement the comprehensive evaluation
            try:
                from training.evaluator import ModelEvaluator
                evaluator = ModelEvaluator(config)
                evaluator.comprehensive_evaluation()
            except ImportError:
                print("Evaluation module not available yet")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()