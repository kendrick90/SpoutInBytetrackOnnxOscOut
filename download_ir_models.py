#!/usr/bin/env python
"""
Download and manage IR-specific detection models
"""

import os
import urllib.request
import logging
import json
import zipfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IRModelDownloader:
    """Download and manage IR-specific detection models"""
    
    def __init__(self):
        self.model_dir = "byte_tracker/model/ir_models"
        self.base_models_dir = "byte_tracker/model"
        self.models_config = {
            "yolov5_thermal_flir": {
                "name": "YOLOv5 FLIR Thermal",
                "description": "YOLOv5s trained on FLIR thermal dataset",
                "source": "vanshksharma/Infrared-Object-Detection",
                "github_repo": "https://github.com/vanshksharma/Infrared-Object-Detection",
                "input_size": "640x640",
                "classes": ["person", "bicycle", "car"],
                "format": "pytorch",
                "notes": "Need to convert to ONNX"
            },
            "yolov5_thermal_multispectral": {
                "name": "YOLOv5 Multispectral Thermal",
                "description": "YOLOv5 for multispectral thermal images",
                "source": "Shaiq1217/YOLOv5Thermal",
                "github_repo": "https://github.com/Shaiq1217/YOLOv5Thermal",
                "input_size": "640x640",
                "classes": ["person", "car", "bicycle", "dog"],
                "format": "pytorch",
                "notes": "FLIR dataset based, need ONNX conversion"
            },
            "yolov8_infrared_lightweight": {
                "name": "YOLOv8 Lightweight IR",
                "description": "Lightweight YOLOv8 for infrared detection",
                "source": "Research paper implementation",
                "input_size": "640x640",
                "classes": ["person", "vehicle"],
                "format": "onnx",
                "notes": "Based on recent research papers"
            }
        }
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def list_available_models(self):
        """List all available IR models"""
        print("\n" + "="*60)
        print("Available IR Detection Models")
        print("="*60)
        
        for i, (key, model) in enumerate(self.models_config.items(), 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Description: {model['description']}")
            print(f"   Source: {model['source']}")
            print(f"   Input Size: {model['input_size']}")
            print(f"   Classes: {', '.join(model['classes'])}")
            print(f"   Format: {model['format']}")
            if 'github_repo' in model:
                print(f"   GitHub: {model['github_repo']}")
            print(f"   Notes: {model['notes']}")
        
        print("\n" + "="*60)
    
    def download_flir_dataset_info(self):
        """Provide information about FLIR dataset"""
        print("\n" + "="*60)
        print("FLIR Thermal Dataset Information")
        print("="*60)
        print("\nOfficial FLIR ADAS Dataset:")
        print("- 14,452 thermal images (gray8 and gray16)")
        print("- 4 object classes: car, person, bicycle, dog")
        print("- California street scenes from vehicle-mounted cameras")
        print("- Free download available")
        print("\nDataset URL:")
        print("https://www.flir.com/oem/adas/adas-dataset-form/")
        print("\nRoboflow FLIR Dataset:")
        print("https://universe.roboflow.com/yolov5-je2ob/flir-rgb/dataset/1")
        print("\n" + "="*60)
    
    def get_conversion_instructions(self):
        """Provide PyTorch to ONNX conversion instructions"""
        print("\n" + "="*60)
        print("Converting PyTorch IR Models to ONNX")
        print("="*60)
        
        print("\n1. Clone the IR model repository:")
        print("   git clone https://github.com/vanshksharma/Infrared-Object-Detection")
        print("   cd Infrared-Object-Detection")
        
        print("\n2. Install YOLOv5 dependencies:")
        print("   pip install ultralytics")
        
        print("\n3. Convert model to ONNX:")
        print("   python -c \"")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('path/to/your/trained_model.pt')")
        print("   model.export(format='onnx', imgsz=640)")
        print("   \"")
        
        print("\n4. Alternative conversion script:")
        print("   python models/export.py --weights best.pt --include onnx")
        
        print("\n5. Copy ONNX model to this project:")
        print(f"   Copy the .onnx file to: {self.model_dir}/")
        
        print("\n" + "="*60)
    
    def create_ir_model_config(self, model_name, model_path):
        """Create configuration for IR model"""
        config = {
            "model_name": model_name,
            "model_path": model_path,
            "is_ir_model": True,
            "input_preprocessing": "none",  # IR models don't need RGB preprocessing
            "classes": self.models_config.get(model_name, {}).get("classes", ["person"]),
            "input_size": self.models_config.get(model_name, {}).get("input_size", "640x640")
        }
        
        config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created config file: {config_path}")
        return config_path
    
    def download_ultralytics_yolov8_model(self):
        """Download a general YOLOv8 model that we can use with IR preprocessing"""
        try:
            from ultralytics import YOLO
            
            print("\nDownloading YOLOv8n model for IR adaptation...")
            model = YOLO('yolov8n.pt')
            
            # Export to ONNX
            onnx_path = model.export(format='onnx', imgsz=640)
            
            # Move to our IR models directory
            target_path = os.path.join(self.model_dir, 'yolov8n_ir_adapted.onnx')
            shutil.move(onnx_path, target_path)
            
            # Create config
            self.create_ir_model_config('yolov8n_ir_adapted', target_path)
            
            print(f"Downloaded and configured: {target_path}")
            print("This model can be used with IR preprocessing")
            
            return target_path
            
        except ImportError:
            print("Ultralytics not installed. Install with: pip install ultralytics")
            return None
        except Exception as e:
            logger.error(f"Failed to download YOLOv8 model: {e}")
            return None


def main():
    downloader = IRModelDownloader()
    
    print("IR Detection Models Manager")
    print("Choose an option:")
    print("1. List available IR models")
    print("2. Get FLIR dataset information")
    print("3. Get PyTorch to ONNX conversion instructions")
    print("4. Download YOLOv8n for IR adaptation")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        downloader.list_available_models()
    elif choice == "2":
        downloader.download_flir_dataset_info()
    elif choice == "3":
        downloader.get_conversion_instructions()
    elif choice == "4":
        model_path = downloader.download_ultralytics_yolov8_model()
        if model_path:
            print(f"\nTo use this model with IR mode:")
            print(f"python SpoutByteTrackOnnxOsc.py --model {model_path} --ir_mode --detector_type yolov8")
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("Next Steps for Best IR Performance:")
    print("="*60)
    print("1. Get IR-specific trained models from GitHub repos listed above")
    print("2. Convert PyTorch models to ONNX format")
    print("3. Train your own model on IR data if needed")
    print("4. Use models with --ir_mode disabled since they're already IR-trained")
    print("5. Test different models to find best performance for your IR camera")


if __name__ == "__main__":
    main()