#!/usr/bin/env python
"""
Test script to demonstrate IR mode capabilities
"""

import argparse
import sys

def main():
    print("=" * 60)
    print("ByteTrack IR Mode Test")
    print("=" * 60)
    
    print("\nAvailable modes:")
    print("1. RGB Mode (default) - Standard color camera")
    print("2. IR Mode - IR illuminated camera with enhancement")
    
    print("\nIR Enhancement methods:")
    print("- histogram: Basic histogram equalization")
    print("- clahe: Adaptive histogram equalization (recommended)")
    print("- gamma: Gamma correction")
    print("- none: No enhancement")
    
    print("\nSupported detector types:")
    print("- auto: Auto-detect from model filename")
    print("- yolox: YOLOX models (ByteTrack default)")
    print("- yolov5: YOLOv5 models")
    print("- yolov7: YOLOv7 models")
    print("- yolov8: YOLOv8 models")
    
    print("\nExample commands:")
    print("\n# Standard RGB mode:")
    print("python SpoutByteTrackOnnxOsc.py")
    
    print("\n# IR mode with CLAHE enhancement:")
    print("python SpoutByteTrackOnnxOsc.py --ir_mode --ir_enhancement clahe")
    
    print("\n# IR mode with specific detector:")
    print("python SpoutByteTrackOnnxOsc.py --ir_mode --detector_type yolov7 --ir_enhancement histogram")
    
    print("\n# Use legacy tracker (original ByteTrackerONNX):")
    print("python SpoutByteTrackOnnxOsc.py --use_legacy_tracker")
    
    print("\nFor IR cameras:")
    print("- Try different enhancement methods to see which works best")
    print("- CLAHE usually works well for IR illuminated scenes")
    print("- You may need models trained on IR/grayscale data for best results")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()