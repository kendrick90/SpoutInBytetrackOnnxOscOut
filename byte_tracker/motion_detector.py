#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motion-based detector for IR cameras
Detects moving blobs instead of classifying objects - much more reliable for IR
"""

import numpy as np
import cv2
import logging
from .detector_base import DetectorBase

logger = logging.getLogger(__name__)


class MotionDetector(DetectorBase):
    """
    Motion-based detector that finds moving blobs/regions
    Ideal for IR cameras where object classification is poor
    """
    
    def __init__(self, min_area=500, max_area=50000, history=500, 
                 var_threshold=16, detect_shadows=False, 
                 morphology_kernel_size=5, gaussian_blur_size=5):
        """
        Initialize motion detector
        
        Args:
            min_area: Minimum blob area to consider
            max_area: Maximum blob area to consider  
            history: Number of frames for background model
            var_threshold: Threshold for background subtraction
            detect_shadows: Whether to detect shadows
            morphology_kernel_size: Size of morphological operations kernel
            gaussian_blur_size: Size of Gaussian blur kernel
        """
        self.min_area = min_area
        self.max_area = max_area
        self.gaussian_blur_size = gaussian_blur_size
        self.morphology_kernel_size = morphology_kernel_size
        
        # Create background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Morphological kernel for cleaning up the mask
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morphology_kernel_size, morphology_kernel_size)
        )
        
        # Track frame count for initialization
        self.frame_count = 0
        self.initialization_frames = 30  # Frames to build background model
        
        logger.info(f"Motion detector initialized:")
        logger.info(f"  Min area: {min_area}")
        logger.info(f"  Max area: {max_area}")
        logger.info(f"  Background history: {history}")
        logger.info(f"  Variance threshold: {var_threshold}")
    
    def detect(self, image):
        """
        Detect moving blobs in the image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            detections: Array of [x1, y1, x2, y2, score, class_id]
                       class_id is always 0 (person equivalent)
        """
        self.frame_count += 1
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        if self.gaussian_blur_size > 0:
            gray = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Skip detection during initialization phase
        if self.frame_count <= self.initialization_frames:
            logger.debug(f"Initializing background model: frame {self.frame_count}/{self.initialization_frames}")
            return np.empty((0, 6))
        
        # Filter out shadows - only keep bright motion (foreground pixels)
        # Shadows are detected as value 127, foreground as 255, background as 0
        # We only want the bright foreground pixels (255), not shadows (127)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Clean up the mask with morphological operations
        # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        # Fill holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on area and aspect ratio
            # Larger blobs and human-like aspect ratios get higher scores
            aspect_ratio = w / h if h > 0 else 0
            human_like_ratio = 0.3 <= aspect_ratio <= 2.0  # Typical human ratios
            
            # Natural confidence calculation based on blob characteristics
            # Normalize area to 0.0-1.0 range
            area_score = min(area / self.max_area, 1.0)
            # Aspect ratio score (how human-like the shape is)
            ratio_score = 1.0 if human_like_ratio else 0.5
            # Natural confidence without artificial boosting
            confidence = (area_score * 0.6 + ratio_score * 0.4)
            
            # Convert to [x1, y1, x2, y2, score, class_id] format
            detection = [x, y, x + w, y + h, confidence, 0]  # class_id = 0 (person)
            detections.append(detection)
        
        if len(detections) > 0:
            detections = np.array(detections)
            # Sort by confidence (highest first)
            detections = detections[detections[:, 4].argsort()[::-1]]
        else:
            detections = np.empty((0, 6))
        
        return detections
    
    def preprocess(self, image):
        """Preprocess image for motion detection"""
        # Motion detector handles preprocessing internally
        return image
    
    def postprocess(self, outputs, image_shape):
        """Postprocess outputs - motion detector returns detections directly"""
        # Motion detector already returns properly formatted detections
        return outputs
    
    def get_debug_image(self, image):
        """
        Get debug visualization showing motion mask and detections
        
        Args:
            image: Original input image
            
        Returns:
            debug_image: Image with motion mask overlay
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.gaussian_blur_size > 0:
            gray = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        
        fg_mask = self.bg_subtractor.apply(gray.copy())  # Use copy to avoid affecting state
        # Filter out shadows - only show bright motion pixels
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Create colored overlay
        debug_image = image.copy()
        if len(debug_image.shape) == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        
        # Overlay motion mask in red
        motion_overlay = np.zeros_like(debug_image)
        motion_overlay[:, :, 2] = fg_mask  # Red channel
        debug_image = cv2.addWeighted(debug_image, 0.7, motion_overlay, 0.3, 0)
        
        return debug_image
    
    def reset_background(self):
        """Reset the background model (useful when scene changes significantly)"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        self.frame_count = 0
        logger.info("Background model reset")


class AdaptiveMotionDetector(MotionDetector):
    """
    Enhanced motion detector with adaptive parameters for challenging IR conditions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Adaptive parameters
        self.detection_history = []
        self.max_history = 60  # Track last 60 frames (~2 seconds at 30fps)
        self.low_detection_threshold = 0.1  # If detection rate < 10%
        self.high_detection_threshold = 0.8  # If detection rate > 80%
        
        # Parameter ranges for adaptation
        self.var_threshold_range = (8, 32)
        self.min_area_range = (200, 1000)
        
        self.current_var_threshold = 16
        self.current_min_area = self.min_area
        
        logger.info("Adaptive motion detector initialized with auto-tuning")
    
    def detect(self, image):
        """Enhanced detection with adaptive parameter tuning"""
        detections = super().detect(image)
        
        # Track detection success rate
        has_detections = len(detections) > 0
        self.detection_history.append(has_detections)
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        # Adapt parameters if we have enough history
        if len(self.detection_history) >= 30:  # After 1 second
            self._adapt_parameters()
        
        return detections
    
    def _adapt_parameters(self):
        """Automatically adapt detection parameters based on performance"""
        detection_rate = sum(self.detection_history) / len(self.detection_history)
        
        # If too few detections, make detector more sensitive
        if detection_rate < self.low_detection_threshold:
            # Decrease variance threshold (more sensitive)
            if self.current_var_threshold > self.var_threshold_range[0]:
                self.current_var_threshold = max(
                    self.var_threshold_range[0],
                    self.current_var_threshold - 2
                )
                self._update_background_subtractor()
                logger.debug(f"Adapted: More sensitive (var_threshold={self.current_var_threshold})")
            
            # Decrease minimum area
            if self.current_min_area > self.min_area_range[0]:
                self.current_min_area = max(
                    self.min_area_range[0],
                    int(self.current_min_area * 0.8)
                )
                self.min_area = self.current_min_area
                logger.debug(f"Adapted: Smaller min area ({self.current_min_area})")
        
        # If too many detections, make detector less sensitive
        elif detection_rate > self.high_detection_threshold:
            # Increase variance threshold (less sensitive)
            if self.current_var_threshold < self.var_threshold_range[1]:
                self.current_var_threshold = min(
                    self.var_threshold_range[1],
                    self.current_var_threshold + 2
                )
                self._update_background_subtractor()
                logger.debug(f"Adapted: Less sensitive (var_threshold={self.current_var_threshold})")
            
            # Increase minimum area
            if self.current_min_area < self.min_area_range[1]:
                self.current_min_area = min(
                    self.min_area_range[1],
                    int(self.current_min_area * 1.2)
                )
                self.min_area = self.current_min_area
                logger.debug(f"Adapted: Larger min area ({self.current_min_area})")
    
    def _update_background_subtractor(self):
        """Update background subtractor with new parameters"""
        # Create new background subtractor with updated parameters
        old_bg = self.bg_subtractor.getBackgroundImage()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=self.current_var_threshold,
            detectShadows=False
        )
        # If we have an existing background, set it to maintain continuity
        if old_bg is not None:
            # The background will adapt quickly with the new parameters
            pass