#!/usr/bin/env python
"""
Download and manage NIR (Near Infrared) detection models
For IR illuminated cameras (850nm/940nm) - not thermal
"""

import os
import urllib.request
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NIRModelDownloader:
    """Download and manage NIR-specific detection models"""
    
    def __init__(self):
        self.model_dir = "byte_tracker/model/nir_models"
        self.models_config = {
            "yolov5_grayscale": {
                "name": "YOLOv5 Grayscale/NIR",
                "description": "YOLOv5 adapted for grayscale/NIR images",
                "wavelength": "850nm/940nm compatible",
                "input_format": "Grayscale or RGB converted from NIR",
                "training_data": "COCO + grayscale augmentation",
                "source": "Standard YOLOv5 with grayscale training",
                "notes": "Good starting point for NIR cameras"
            },
            "yolov8_nighttime": {
                "name": "YOLOv8 Nighttime Detection", 
                "description": "YOLOv8 optimized for low-light/nighttime",
                "wavelength": "NIR compatible",
                "input_format": "Grayscale/NIR images",
                "training_data": "Mixed daylight/nighttime datasets",
                "source": "Research implementations",
                "notes": "Better for actual NIR surveillance footage"
            },
            "darknet_lowlight": {
                "name": "Darknet Low-light Detection",
                "description": "Darknet/YOLO variants for low-light conditions",
                "wavelength": "850nm/940nm optimized",
                "input_format": "Single channel or converted NIR",
                "training_data": "Low-light surveillance datasets",
                "source": "Academic research projects",
                "notes": "Specialized for surveillance applications"
            }
        }
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def list_nir_models(self):
        """List available NIR detection models"""
        print("\n" + "="*70)
        print("NIR (Near Infrared) Detection Models for IR Illuminated Cameras")
        print("="*70)
        print("For 850nm/940nm IR LED illuminated cameras (NOT thermal)")
        print()
        
        for i, (key, model) in enumerate(self.models_config.items(), 1):
            print(f"{i}. {model['name']}")
            print(f"   Description: {model['description']}")
            print(f"   Wavelength: {model['wavelength']}")
            print(f"   Input: {model['input_format']}")
            print(f"   Training: {model['training_data']}")
            print(f"   Source: {model['source']}")
            print(f"   Notes: {model['notes']}")
            print()
        
        print("="*70)
    
    def get_nir_camera_info(self):
        """Provide info about NIR vs thermal cameras"""
        print("\n" + "="*70)
        print("NIR Illuminated vs Thermal Cameras")
        print("="*70)
        
        print("\nNIR Illuminated Cameras (What you have):")
        print("- Wavelength: 850nm or 940nm LEDs")
        print("- Image: Grayscale, looks like B&W photos")
        print("- Use: Night vision, surveillance")
        print("- Detection: Can use RGB models with grayscale conversion")
        print("- Best models: Trained on grayscale/nighttime data")
        
        print("\nThermal Cameras (Different technology):")
        print("- Wavelength: 8-14μm (much longer)")
        print("- Image: Heat signatures, false color")
        print("- Use: Temperature detection, medical")
        print("- Detection: Needs thermal-specific trained models")
        
        print("\nFor your NIR cameras:")
        print("1. Standard YOLO models work reasonably well")
        print("2. Convert images to grayscale or use all 3 channels")
        print("3. Models trained on nighttime/low-light data work better")
        print("4. 850nm generally gives better detection than 940nm")
        
        print("\n" + "="*70)
    
    def get_model_recommendations(self):
        """Provide specific model recommendations for NIR"""
        print("\n" + "="*70)
        print("Model Recommendations for NIR Cameras")
        print("="*70)
        
        print("\n🥇 BEST APPROACH - Use existing RGB models:")
        print("1. Your current ByteTrack/YOLOX model should work")
        print("2. NIR images are just grayscale - YOLO can handle this")
        print("3. Disable IR enhancement - not needed for NIR")
        print("4. Command: python SpoutByteTrackOnnxOsc.py --ir_mode")
        
        print("\n🥈 BETTER PERFORMANCE - Nighttime trained models:")
        print("1. YOLOv8 models trained on nighttime datasets")
        print("2. Models trained with grayscale augmentation")
        print("3. Low-light surveillance specific models")
        
        print("\n🏆 OPTIMAL - Custom training:")
        print("1. Fine-tune existing YOLO on your NIR footage")
        print("2. Use datasets like:")
        print("   - ExDark (low-light detection)")
        print("   - Dark Face (nighttime face detection)")
        print("   - LLVIP (Low Light Vision datasets)")
        
        print("\nImmediate steps to try:")
        print("1. Test current model: python SpoutByteTrackOnnxOsc.py")
        print("2. Try with IR mode: python SpoutByteTrackOnnxOsc.py --ir_mode --ir_enhancement none")
        print("3. If poor results, try contrast enhancement")
        
        print("\n" + "="*70)
    
    def download_yolov8_for_nir(self):
        """Download standard YOLOv8 for NIR adaptation"""
        try:
            from ultralytics import YOLO
            
            print("\nDownloading YOLOv8n for NIR adaptation...")
            model = YOLO('yolov8n.pt')
            
            # Export to ONNX
            onnx_path = model.export(format='onnx', imgsz=640)
            
            # Move to NIR models directory
            import shutil
            target_path = os.path.join(self.model_dir, 'yolov8n_nir.onnx')
            shutil.move(onnx_path, target_path)
            
            # Create config
            config = {
                "model_name": "yolov8n_nir",
                "model_path": target_path,
                "is_nir_model": False,  # It's an RGB model used for NIR
                "preprocessing": "grayscale_conversion",
                "wavelength": "850nm/940nm compatible",
                "notes": "Standard YOLOv8n adapted for NIR use"
            }
            
            config_path = os.path.join(self.model_dir, 'yolov8n_nir_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Downloaded: {target_path}")
            print(f"Config: {config_path}")
            print(f"\nTo use: python SpoutByteTrackOnnxOsc.py --model {target_path} --detector_type yolov8")
            
            return target_path
            
        except ImportError:
            print("Ultralytics not installed. Install with: pip install ultralytics")
            return None
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            return None
    
    def test_current_model_for_nir(self):
        """Test current model performance suggestions"""
        print("\n" + "="*70)
        print("Testing Your Current Model with NIR")
        print("="*70)
        
        print("\nStep 1 - Test without modifications:")
        print("python SpoutByteTrackOnnxOsc.py")
        print("(Your current ByteTrack model might work fine)")
        
        print("\nStep 2 - If detection is poor, try basic enhancement:")
        print("python SpoutByteTrackOnnxOsc.py --ir_mode --ir_enhancement clahe")
        
        print("\nStep 3 - Try different enhancement methods:")
        print("python SpoutByteTrackOnnxOsc.py --ir_mode --ir_enhancement histogram")
        print("python SpoutByteTrackOnnxOsc.py --ir_mode --ir_enhancement gamma")
        
        print("\nStep 4 - Disable enhancement (NIR might not need it):")
        print("python SpoutByteTrackOnnxOsc.py --ir_mode --ir_enhancement none")
        
        print("\nWhat to look for:")
        print("✅ Good: People detected and tracked consistently")
        print("❌ Bad: Missing detections, false positives")
        print("🔧 Fix: Try different enhancement methods or get NIR-trained model")
        
        print("\nNIR camera tips:")
        print("- 850nm LEDs: Usually better detection performance")
        print("- 940nm LEDs: More covert but ~30% less effective")
        print("- Ensure good IR illumination in the scene")
        print("- Distance affects IR illumination effectiveness")
        
        print("\n" + "="*70)


def main():
    downloader = NIRModelDownloader()
    
    print("NIR Detection Models Manager")
    print("For IR Illuminated Cameras (850nm/940nm)")
    print()
    print("Choose an option:")
    print("1. List available NIR models")
    print("2. NIR vs Thermal camera info")
    print("3. Model recommendations for NIR cameras")
    print("4. Download YOLOv8n for NIR adaptation")
    print("5. Test current model with NIR")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        downloader.list_nir_models()
    elif choice == "2":
        downloader.get_nir_camera_info()
    elif choice == "3":
        downloader.get_model_recommendations()
    elif choice == "4":
        downloader.download_yolov8_for_nir()
    elif choice == "5":
        downloader.test_current_model_for_nir()
    elif choice == "6":
        print("Exiting...")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()