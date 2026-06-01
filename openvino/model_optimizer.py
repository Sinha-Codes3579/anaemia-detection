import torch
import openvino as ov
import nncf
import os
from openvino.runtime import Core
import numpy as np

class OpenVINOModelOptimizer:
    def __init__(self, config):
        self.config = config
        self.core = Core()
    
    def convert_to_onnx(self, model, dummy_input, onnx_path):
        """Convert PyTorch model to ONNX"""
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model converted to ONNX: {onnx_path}")
    
    def convert_to_openvino(self, onnx_path, openvino_path):
        """Convert ONNX model to OpenVINO IR"""
        model = self.core.read_model(onnx_path)
        
        # Optimize for CPU
        compiled_model = ov.compile_model(model, "CPU")
        
        # Save the model
        ov.save_model(compiled_model, openvino_path)
        print(f"Model converted to OpenVINO: {openvino_path}")
        
        return compiled_model
    
    def quantize_model(self, model, calibration_loader):
        """Quantize model for better performance"""
        def transform_fn(data_item):
            images, _ = data_item
            return images.numpy()
        
        # Create calibration dataset
        calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
        
        # Quantize the model
        quantized_model = nncf.quantize(model, calibration_dataset)
        
        quantized_path = self.config.OPENVINO_MODEL_PATH.replace('.xml', '_quantized.xml')
        ov.save_model(quantized_model, quantized_path)
        print(f"Quantized model saved: {quantized_path}")
        
        return quantized_model
    
    def optimize_model(self, pytorch_model, calibration_loader=None):
        """Full optimization pipeline"""
        # Create dummy input
        dummy_input = torch.randn(1, 3, *self.config.IMAGE_SIZE)
        
        # Convert to ONNX
        onnx_path = self.config.OPENVINO_MODEL_PATH.replace('.xml', '.onnx')
        self.convert_to_onnx(pytorch_model, dummy_input, onnx_path)
        
        # Convert to OpenVINO
        openvino_model = self.convert_to_openvino(onnx_path, self.config.OPENVINO_MODEL_PATH)
        
        # Quantize if calibration data is provided
        if calibration_loader:
            quantized_model = self.quantize_model(openvino_model, calibration_loader)
            return quantized_model
        
        return openvino_model

class OpenVINOPredictor:
    def __init__(self, config, model_path):
        self.config = config
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        
        # Get input and output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
    
    def preprocess_image(self, image):
        """Preprocess image for OpenVINO inference"""
        import cv2
        import numpy as np
        
        # Resize
        image = cv2.resize(image, self.config.IMAGE_SIZE)
        
        # Normalize
        image = image.astype(np.float32)
        image = image / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Change to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, 0)
        
        return image
    
    def predict(self, image):
        """Perform inference using OpenVINO"""
        processed_image = self.preprocess_image(image)
        
        # Inference
        result = self.compiled_model([processed_image])[self.output_layer]
        
        # Post-process
        probabilities = torch.softmax(torch.from_numpy(result), dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        return {
            'class': self.config.CLASS_NAMES[prediction.item()],
            'confidence': float(probabilities[0][prediction.item()]),
            'probabilities': {
                self.config.CLASS_NAMES[i]: float(prob) 
                for i, prob in enumerate(probabilities.numpy()[0])
            }
        }