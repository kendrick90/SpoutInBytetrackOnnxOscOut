#!/usr/bin/env python
"""
Download and convert YOLO-IR models for infrared person detection
"""

import os
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOIRModelManager:
    """Manage YOLO-IR model downloads and conversions"""
    
    def __init__(self):
        self.models_dir = "byte_tracker/model/yolo_ir"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def list_available_models(self):
        """Show available YOLO-IR models"""
        print("\n" + "="*70)
        print("Available YOLO-IR Models for Infrared Person Detection")
        print("="*70)
        
        models = {
            "flir_yolov5": {
                "name": "YOLOv5 FLIR Thermal",
                "source": "vanshksharma/Infrared-Object-Detection",
                "github": "https://github.com/vanshksharma/Infrared-Object-Detection",
                "description": "YOLOv5 trained on infrared images for person/bicycle/car detection",
                "classes": ["person", "bicycle", "car"],
                "format": "PyTorch (.pt) - needs conversion to ONNX",
                "trained_on": "Infrared/thermal images"
            },
            "yolo_ir_free": {
                "name": "YOLO-IR-Free",
                "source": "Research paper implementation",
                "description": "Improved YOLO for infrared vehicle detection with TSE attention",
                "classes": ["vehicle", "person"],
                "format": "Research code - implementation needed",
                "trained_on": "Infrared vehicle datasets"
            },
            "flir_yolov8": {
                "name": "YOLOv8 FLIR Custom",
                "source": "Custom training on FLIR dataset",
                "description": "YOLOv8 trained on FLIR thermal dataset",
                "classes": ["person", "bicycle", "car", "dog"],
                "format": "Train yourself using FLIR dataset",
                "trained_on": "FLIR thermal dataset (14,452 images)"
            }
        }
        
        for i, (key, model) in enumerate(models.items(), 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Description: {model['description']}")
            print(f"   Classes: {', '.join(model['classes'])}")
            print(f"   Format: {model['format']}")
            print(f"   Trained on: {model['trained_on']}")
            if 'github' in model:
                print(f"   GitHub: {model['github']}")
        
        print("\n" + "="*70)
    
    def get_flir_yolov5_instructions(self):
        """Instructions to get YOLOv5 FLIR model"""
        print("\n" + "="*70)
        print("Get YOLOv5 FLIR Thermal Model")
        print("="*70)
        
        print("\nThis model is trained on infrared images for person detection.")
        print("GitHub: https://github.com/vanshksharma/Infrared-Object-Detection")
        
        print("\nSteps to get the model:")
        print("1. Clone the repository:")
        print("   git clone https://github.com/vanshksharma/Infrared-Object-Detection")
        print("   cd Infrared-Object-Detection")
        
        print("\n2. The trained weights are at:")
        print("   yolov5/runs/train/Result/weights/best.pt")
        
        print("\n3. Convert to ONNX:")
        print("   cd yolov5")
        print("   python export.py --weights runs/train/Result/weights/best.pt --include onnx")
        
        print("\n4. Copy ONNX model:")
        print(f"   Copy the .onnx file to: {self.models_dir}/")
        
        print("\n5. Use with ByteTrack:")
        print("   python SpoutByteTrackOnnxOsc.py --model byte_tracker/model/yolo_ir/best.onnx --detector_type yolo_ir")
        
        print("\n" + "="*70)
    
    def get_yolo_ir_free_info(self):
        """Information about YOLO-IR-Free"""
        print("\n" + "="*70)
        print("YOLO-IR-Free: Advanced IR Detection")
        print("="*70)
        
        print("\nYOLO-IR-Free is a research paper implementation:")
        print("Title: 'YOLO-IR-Free: An Improved Algorithm for Real-Time Detection of Vehicles in Infrared Images'")
        print("Published: MDPI Sensors 2023")
        print("URL: https://www.mdpi.com/1424-8220/23/21/8723")
        
        print("\nKey improvements:")
        print("- TSE (Thermal Spatial Enhancement) attention mechanism")
        print("- Rep-LAN (Reparameterized Large Kernel Attention Network)")
        print("- Optimized for infrared vehicle detection")
        
        print("\nTo implement:")
        print("1. Read the research paper for architecture details")
        print("2. Implement the TSE attention mechanism")
        print("3. Train on infrared vehicle datasets")
        print("4. Export to ONNX format")
        
        print("\nAlternatively:")
        print("- Contact the authors for pre-trained weights")
        print("- Look for community implementations on GitHub")
        
        print("\n" + "="*70)
    
    def train_flir_yolov8_instructions(self):
        """Instructions to train YOLOv8 on FLIR dataset"""
        print("\n" + "="*70)
        print("Train YOLOv8 on FLIR Thermal Dataset")
        print("="*70)
        
        print("\nFLIR Thermal Dataset:")
        print("- 14,452 thermal images (gray8 and gray16)")
        print("- 4 classes: person, bicycle, car, dog")
        print("- Free download: https://www.flir.com/oem/adas/adas-dataset-form/")
        
        print("\nTraining steps:")
        print("1. Download FLIR dataset")
        print("2. Convert annotations to YOLO format")
        print("3. Install Ultralytics: pip install ultralytics")
        print("4. Train YOLOv8:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('yolov8n.pt')")
        print("   model.train(data='flir.yaml', epochs=100)")
        
        print("\n5. Export to ONNX:")
        print("   model.export(format='onnx')")
        
        print("\n6. Use with ByteTrack:")
        print("   python SpoutByteTrackOnnxOsc.py --model path/to/yolov8_flir.onnx --detector_type yolo_ir")
        
        print("\nPre-processed FLIR datasets:")
        print("- Roboflow: https://universe.roboflow.com/yolov5-je2ob/flir-rgb/dataset/1")
        print("- GitHub expansions with more annotations available")
        
        print("\n" + "="*70)
    
    def download_yolov8_for_ir_training(self):
        """Download base YOLOv8 model for IR training"""
        try:
            # Install ultralytics if not installed
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            
            from ultralytics import YOLO
            
            print("\nDownloading YOLOv8n for IR training...")
            model = YOLO('yolov8n.pt')
            
            print("YOLOv8n downloaded successfully!")
            print("You can now train this on FLIR dataset or other IR datasets.")
            print("\nNext steps:")
            print("1. Get FLIR dataset: https://www.flir.com/oem/adas/adas-dataset-form/")
            print("2. Train: model.train(data='your_ir_dataset.yaml', epochs=100)")
            print("3. Export: model.export(format='onnx')")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download YOLOv8: {e}")
            print("Manual installation:")
            print("pip install ultralytics")
            print("python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
            return False
    
    def quick_test_setup(self):
        """Quick setup for testing with existing models"""
        print("\n" + "="*70)
        print("Quick Test Setup")
        print("="*70)
        
        print("\nFor immediate testing with your current model:")
        print("1. Use existing ByteTrack model:")
        print("   python SpoutByteTrackOnnxOsc.py --detector_type yolo_ir")
        
        print("\n2. This will:")
        print("   - Use your current model but with IR-optimized detector")
        print("   - Handle grayscale IR images properly")
        print("   - Skip RGB preprocessing")
        
        print("\nIf performance is poor:")
        print("1. Get a model trained on IR data (see options above)")
        print("2. Train your own on FLIR dataset")
        print("3. Try different detection thresholds:")
        print("   --score_th 0.1 --nms_th 0.5")
        
        print("\n" + "="*70)


def main():
    manager = YOLOIRModelManager()
    
    print("YOLO-IR Model Manager")
    print("For Infrared Person Detection")
    print()
    print("Choose an option:")
    print("1. List available YOLO-IR models")
    print("2. Get YOLOv5 FLIR model (trained on IR data)")
    print("3. YOLO-IR-Free information")
    print("4. Train YOLOv8 on FLIR dataset")
    print("5. Download YOLOv8 for IR training")
    print("6. Quick test setup")
    print("7. Exit")
    
    try:
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            manager.list_available_models()
        elif choice == "2":
            manager.get_flir_yolov5_instructions()
        elif choice == "3":
            manager.get_yolo_ir_free_info()
        elif choice == "4":
            manager.train_flir_yolov8_instructions()
        elif choice == "5":
            manager.download_yolov8_for_ir_training()
        elif choice == "6":
            manager.quick_test_setup()
        elif choice == "7":
            print("Exiting...")
        else:
            print("Invalid choice")
    except EOFError:
        print("\nRunning in non-interactive mode. Use option 6 for quick setup.")
        manager.quick_test_setup()


if __name__ == "__main__":
    main()