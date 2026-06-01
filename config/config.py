import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    # Project paths
    PROJECT_ROOT: str = "/content/drive/MyDrive/anaemiadetect/Anaemia"
    # DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data_fixed")
    MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models")
    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
    RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
    YOLO_DIR: str = os.path.join(PROJECT_ROOT, "yolov8")
    OPENVINO_DIR: str = os.path.join(PROJECT_ROOT, "openvino")
    
    # Data configuration
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # Model configuration
    NUM_CLASSES: int = 2
    LEARNING_RATE: float = 0.0003
    NUM_EPOCHS: int = 50
    WEIGHT_DECAY: float = 5e-4
    USE_FEATURES: bool = True
    
    # Classes - FIXED: using default_factory for mutable list
    CLASS_NAMES: list = field(default_factory=lambda: ["anaemic", "non_anaemic"])
    
    # Training configuration
    EARLY_STOPPING_PATIENCE: int = 20
    MODEL_SAVE_PATH: str = os.path.join(MODEL_DIR, "anemia_model.pth")
    # YOLOv8 Configuration
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    YOLO_CUSTOM_MODEL_PATH: str = os.path.join(YOLO_DIR, "models", "nail_detector.pt")
    YOLO_CONFIDENCE: float = 0.5
    YOLO_EPOCHS: int = 100
    YOLO_IMAGE_SIZE: int = 640
    
    # OpenVINO Configuration
    OPENVINO_MODEL_PATH: str = os.path.join(OPENVINO_DIR, "models", "anemia_model.xml")
    
    def __post_init__(self):
      directories = [
            self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR, self.RESULTS_DIR,
            self.YOLO_DIR, os.path.join(self.YOLO_DIR, "models"),
            os.path.join(self.YOLO_DIR, "detection_results"),
            self.OPENVINO_DIR, os.path.join(self.OPENVINO_DIR, "models"),
            os.path.join(self.OPENVINO_DIR, "optimized")
      ]
        
      for directory in directories:
            os.makedirs(directory, exist_ok=True)