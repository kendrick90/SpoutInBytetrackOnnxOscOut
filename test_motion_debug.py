#!/usr/bin/env python
"""
Debug motion detection to see what's happening
"""

import cv2
import numpy as np
import time
import array
from itertools import repeat
import SpoutGL
from OpenGL.GL import *
from byte_tracker.motion_detector import MotionDetector

def main():
    print("Motion Detection Debug Tool")
    print("This will show you what the motion detector sees")
    
    # Create motion detector with very sensitive settings
    detector = MotionDetector(
        min_area=50,
        max_area=100000,
        var_threshold=8,
        history=200,  # Shorter history for faster adaptation
        detect_shadows=False
    )
    
    print(f"Motion detector settings:")
    print(f"  Min area: {detector.min_area}")
    print(f"  Max area: {detector.max_area}")
    print(f"  Variance threshold: 8")
    print(f"  History: 200 frames")
    
    # Set up Spout receiver
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)
    
    with SpoutGL.SpoutReceiver() as receiver:
        receiver.setReceiverName("TouchDesigner")
        print("Waiting for Spout sender 'TouchDesigner'...")
        
        buffer = None
        frame_count = 0
        
        while True:
            # Get frame from Spout
            result = receiver.receiveImage(buffer, GL_RGB, False, 0)
            
            # Resize if sender changes size
            if receiver.isUpdated():
                width = receiver.getSenderWidth()
                height = receiver.getSenderHeight()
                buffer = array.array('B', repeat(0, width * height * 3))
                print(f"Spout sender detected! Size: {width}x{height}")
            
            if buffer and result and not SpoutGL.helpers.isBufferEmpty(buffer):
                frame_count += 1
                
                # Convert buffer to image
                frame = np.array(buffer)
                frame = np.reshape(frame, (height, width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Show original frame
                cv2.imshow('Original', frame)
                
                # Get motion mask for debugging
                debug_image = detector.get_debug_image(frame)
                cv2.imshow('Debug View', debug_image)
                
                # Get raw motion mask
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if detector.gaussian_blur_size > 0:
                    gray = cv2.GaussianBlur(gray, (detector.gaussian_blur_size, detector.gaussian_blur_size), 0)
                
                fg_mask = detector.bg_subtractor.apply(gray.copy())
                
                # Show motion mask before morphology
                cv2.imshow('Motion Mask', fg_mask)
                
                # Run detection
                detections = detector.detect(frame)
                
                # Print debug info
                if frame_count % 30 == 0:  # Every second at 30fps
                    print(f"Frame {frame_count}: {len(detections)} detections")
                    if len(detections) > 0:
                        for i, det in enumerate(detections):
                            x1, y1, x2, y2, score, class_id = det
                            print(f"  Detection {i}: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) score={score:.3f}")
                    
                    # Count motion pixels
                    motion_pixels = np.sum(fg_mask > 0)
                    total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
                    motion_percent = (motion_pixels / total_pixels) * 100
                    print(f"  Motion pixels: {motion_pixels}/{total_pixels} ({motion_percent:.2f}%)")
                    
                    # Check if background is still initializing
                    if frame_count <= detector.initialization_frames:
                        print(f"  Still initializing background: {frame_count}/{detector.initialization_frames}")
                
                # Draw detections on original
                display_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2, score, class_id = det
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{score:.2f}", (int(x1), int(y1-5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Update original window with detections
                cv2.imshow('Original', display_frame)
                
            else:
                # No input
                if frame_count == 0:
                    print("No Spout input. Make sure TouchDesigner is running and sending.")
                time.sleep(0.1)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("ESC pressed, exiting...")
                break
            elif key == ord('r'):
                print("Resetting background model...")
                detector.reset_background()
                frame_count = 0
            elif key == ord('s'):
                print("Saving current frame and mask...")
                if 'frame' in locals():
                    cv2.imwrite('debug_original.jpg', frame)
                    cv2.imwrite('debug_mask.jpg', fg_mask)
                    cv2.imwrite('debug_overlay.jpg', debug_image)
                    print("Saved: debug_original.jpg, debug_mask.jpg, debug_overlay.jpg")
    
    cv2.destroyAllWindows()
    print("Motion detection debug finished")

if __name__ == "__main__":
    main()