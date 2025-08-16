#!/usr/bin/env python
"""
Download or convert ByteTrack models
Provides options for pre-converted ONNX models or conversion instructions
"""

import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """Download a file with progress reporting"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
        print(f"\rDownload progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
    
    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {destination}")
    try:
        urllib.request.urlretrieve(url, destination, reporthook=download_progress)
        print()  # New line after progress
        logger.info("Download complete!")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("ByteTrack Model Download/Conversion Helper")
    print("="*60)
    
    # Known PyTorch checkpoint URLs
    pytorch_models = {
        "bytetrack_s_mot17": {
            "url": "https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing",
            "input_size": "608x1088",
            "description": "ByteTrack-S trained on MOT17 (Small)",
            "direct": False
        },
        "bytetrack_m_mot17": {
            "url": "https://drive.google.com/file/d/11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun/view?usp=sharing",
            "input_size": "800x1440", 
            "description": "ByteTrack-M trained on MOT17 (Medium)",
            "direct": False
        },
        "bytetrack_l_mot17": {
            "url": "https://drive.google.com/file/d/1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz/view?usp=sharing",
            "input_size": "1088x1920",
            "description": "ByteTrack-L trained on MOT17 (Large)",
            "direct": False
        },
        "bytetrack_x_mot17": {
            "url": "https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing",
            "input_size": "1088x1920",
            "description": "ByteTrack-X trained on MOT17 (Extra Large)",
            "direct": False
        }
    }
    
    # Pre-converted ONNX models (if available from community)
    onnx_models = {
        "yolox_s_bytetrack": {
            "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
            "input_size": "640x640",
            "description": "YOLOX-S model (can be used with ByteTrack)",
            "direct": True
        }
    }
    
    print("\nOptions:")
    print("1. Download pre-converted ONNX models (limited availability)")
    print("2. Get PyTorch model links for manual conversion")
    print("3. Instructions for converting existing models to ONNX")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("\nAvailable ONNX models:")
        for i, (name, info) in enumerate(onnx_models.items(), 1):
            print(f"{i}. {info['description']}")
            print(f"   Input size: {info['input_size']}")
        
        model_choice = input(f"\nSelect model (1-{len(onnx_models)}): ").strip()
        try:
            idx = int(model_choice) - 1
            model_name = list(onnx_models.keys())[idx]
            model_info = onnx_models[model_name]
            
            model_dir = "byte_tracker/model"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            destination = os.path.join(model_dir, f"{model_name}.onnx")
            
            if model_info["direct"]:
                if download_file(model_info["url"], destination):
                    print(f"\nModel saved to: {destination}")
                    print(f"Run with: python SpoutByteTrackOnnxOsc.py --model {destination}")
            else:
                print(f"\nThis model requires manual download from:")
                print(model_info["url"])
                
        except (ValueError, IndexError):
            print("Invalid selection")
            
    elif choice == "2":
        print("\nPyTorch ByteTrack Models (requires manual download and conversion):")
        print("\nNote: These are Google Drive links that require manual download")
        print("="*60)
        for name, info in pytorch_models.items():
            print(f"\n{info['description']}:")
            print(f"  Download: {info['url']}")
            print(f"  Input size: {info['input_size']}")
            print(f"  Filename: {name}.pth.tar")
        
        print("\n" + "="*60)
        print("\nTo convert PyTorch models to ONNX:")
        print("1. Clone ByteTrack repo: git clone https://github.com/FoundationVision/ByteTrack.git")
        print("2. Download the .pth.tar file from the link above")
        print("3. Run conversion:")
        print("   python tools/export_onnx.py --output-name model.onnx \\")
        print("          -f exps/example/mot/yolox_s_mix_det.py \\")
        print("          -c path/to/downloaded.pth.tar")
        
    elif choice == "3":
        print("\n" + "="*60)
        print("Converting ByteTrack Models to ONNX")
        print("="*60)
        print("\n1. Clone the ByteTrack repository:")
        print("   git clone https://github.com/FoundationVision/ByteTrack.git")
        print("   cd ByteTrack")
        print("\n2. Install requirements:")
        print("   pip install -r requirements.txt")
        print("   pip install onnx onnxsim")
        print("\n3. Download a pre-trained model (see option 2 for links)")
        print("\n4. Export to ONNX:")
        print("   For ByteTrack-S (384x640 or 608x1088):")
        print("   python tools/export_onnx.py --output-name bytetrack_s.onnx \\")
        print("          -f exps/example/mot/yolox_s_mix_det.py \\")
        print("          -c bytetrack_s_mot17.pth.tar \\")
        print("          --input_shape 384,640")
        print("\n   For other models, adjust the exp file and input shape accordingly")
        print("\n5. Copy the ONNX file to this project's byte_tracker/model/ directory")
        print("\n6. Run with: python SpoutByteTrackOnnxOsc.py --model byte_tracker/model/your_model.onnx")
    
    else:
        print("Invalid option")
    
    print("\n" + "="*60)
    print("Alternative: Using the existing model")
    print("="*60)
    print("\nThe current bytetrack_s.onnx model in your folder should work.")
    print("The code has been updated to auto-detect the model's input size.")
    print("Just run: python SpoutByteTrackOnnxOsc.py")
    print("\nThe model will automatically resize inputs to match its requirements.")

if __name__ == "__main__":
    main()