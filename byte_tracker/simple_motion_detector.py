#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple frame differencing motion detector
Sometimes works better than background subtraction for certain setups
"""

import numpy as np
import cv2
import logging
from .detector_base import DetectorBase

logger = logging.getLogger(__name__)


class SimpleMotionDetector(DetectorBase):
    """
    Simple frame differencing motion detector
    Compares consecutive frames to detect motion
    """
    
    def __init__(self, min_area=300, max_area=50000, diff_threshold=25, 
                 blur_size=5, morph_kernel_size=5):
        """
        Initialize simple motion detector
        
        Args:
            min_area: Minimum blob area to consider
            max_area: Maximum blob area to consider
            diff_threshold: Threshold for frame difference
            blur_size: Gaussian blur size
            morph_kernel_size: Morphological operations kernel size
        """
        self.min_area = min_area
        self.max_area = max_area
        self.diff_threshold = diff_threshold
        self.blur_size = blur_size
        
        # Store previous frames for comparison
        self.prev_frame = None
        self.prev_prev_frame = None
        
        # Morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        self.frame_count = 0
        
        logger.info(f"Simple motion detector initialized:")
        logger.info(f"  Min area: {min_area}")
        logger.info(f"  Max area: {max_area}")
        logger.info(f"  Difference threshold: {diff_threshold}")
        logger.info(f"  Blur size: {blur_size}")
    
    def detect(self, image):
        """
        Detect motion using frame differencing
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            detections: Array of [x1, y1, x2, y2, score, class_id]
        """
        self.frame_count += 1
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply blur to reduce noise
        if self.blur_size > 0:
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # Need at least 2 frames to compute difference
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return np.empty((0, 6))
        
        # Compute frame difference
        diff = cv2.absdiff(gray, self.prev_frame)
        
        # If we have 3 frames, use double difference for better results
        if self.prev_prev_frame is not None:
            diff2 = cv2.absdiff(self.prev_frame, self.prev_prev_frame)
            # Combine differences
            diff = cv2.bitwise_and(diff, diff2)
        
        # Threshold the difference
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on area and difference intensity
            roi_diff = diff[y:y+h, x:x+w]
            avg_intensity = np.mean(roi_diff)
            
            # Natural confidence calculation based on motion characteristics
            # Normalize area to 0.0-1.0 range
            area_score = min(area / self.max_area, 1.0)
            # Motion intensity score
            intensity_score = min(avg_intensity / 255.0, 1.0)
            # Natural confidence without artificial boosting
            confidence = (area_score * 0.5 + intensity_score * 0.5)
            
            # Convert to [x1, y1, x2, y2, score, class_id] format
            detection = [x, y, x + w, y + h, confidence, 0]  # class_id = 0 (person)
            detections.append(detection)
        
        # Update frame history
        self.prev_prev_frame = self.prev_frame.copy() if self.prev_frame is not None else None
        self.prev_frame = gray.copy()
        
        if len(detections) > 0:
            detections = np.array(detections)
            # Sort by confidence (highest first)
            detections = detections[detections[:, 4].argsort()[::-1]]
        else:
            detections = np.empty((0, 6))
        
        return detections
    
    def preprocess(self, image):
        """Preprocess image for motion detection"""
        return image
    
    def postprocess(self, outputs, image_shape):
        """Postprocess outputs"""
        return outputs
    
    def get_debug_image(self, image):
        """Get debug visualization"""
        if self.prev_frame is None:
            return image
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.blur_size > 0:
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # Compute difference
        diff = cv2.absdiff(gray, self.prev_frame)
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        
        # Create colored overlay
        debug_image = image.copy()
        if len(debug_image.shape) == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        
        # Overlay motion in green
        motion_overlay = np.zeros_like(debug_image)
        motion_overlay[:, :, 1] = thresh  # Green channel
        debug_image = cv2.addWeighted(debug_image, 0.7, motion_overlay, 0.3, 0)
        
        return debug_image
    
    def reset(self):
        """Reset the detector state"""
        self.prev_frame = None
        self.prev_prev_frame = None
        self.frame_count = 0
        logger.info("Simple motion detector reset")


class HybridMotionDetector(DetectorBase):
    """
    Hybrid motion detector that combines background subtraction and frame differencing
    Uses the best of both approaches
    """
    
    def __init__(self, **kwargs):
        # Import here to avoid circular imports
        from .motion_detector import MotionDetector
        
        self.bg_detector = MotionDetector(**kwargs)
        self.diff_detector = SimpleMotionDetector(**kwargs)
        
        logger.info("Hybrid motion detector initialized (BG subtraction + frame diff)")
    
    def detect(self, image):
        """Detect using both methods and combine results"""
        # Get detections from both methods
        bg_detections = self.bg_detector.detect(image)
        diff_detections = self.diff_detector.detect(image)
        
        # Combine detections (simple concatenation for now)
        if len(bg_detections) > 0 and len(diff_detections) > 0:
            combined = np.vstack([bg_detections, diff_detections])
        elif len(bg_detections) > 0:
            combined = bg_detections
        elif len(diff_detections) > 0:
            combined = diff_detections
        else:
            combined = np.empty((0, 6))
        
        # Remove overlapping detections using NMS-like approach
        if len(combined) > 1:
            combined = self._remove_overlaps(combined)
        
        return combined
    
    def _remove_overlaps(self, detections, iou_threshold=0.3):
        """Remove overlapping detections"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        sorted_idx = np.argsort(detections[:, 4])[::-1]
        keep = []
        
        while len(sorted_idx) > 0:
            # Keep the highest confidence detection
            current_idx = sorted_idx[0]
            keep.append(current_idx)
            
            if len(sorted_idx) == 1:
                break
            
            # Calculate IoU with remaining detections
            current_box = detections[current_idx, :4]
            remaining_boxes = detections[sorted_idx[1:], :4]
            
            ious = self._calculate_iou(current_box, remaining_boxes)
            
            # Remove detections with high IoU
            sorted_idx = sorted_idx[1:][ious < iou_threshold]
        
        return detections[keep]
    
    def _calculate_iou(self, box, boxes):
        """Calculate IoU between one box and multiple boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box_area + boxes_area - intersection
        
        return intersection / (union + 1e-6)
    
    def preprocess(self, image):
        return image
    
    def postprocess(self, outputs, image_shape):
        return outputs
    
    def get_debug_image(self, image):
        """Get debug image from background subtraction method"""
        return self.bg_detector.get_debug_image(image)
    
    def reset(self):
        """Reset both detectors"""
        self.bg_detector.reset_background()
        self.diff_detector.reset()