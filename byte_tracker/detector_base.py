#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for object detectors
Allows swapping between different detection models
"""

from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DetectorBase(ABC):
    """Abstract base class for object detectors"""
    
    def __init__(self, model_path, input_shape=None, providers=None):
        self.model_path = model_path
        self.input_shape = input_shape
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
    @abstractmethod
    def detect(self, image):
        """
        Perform object detection on an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            detections: Array of detections [x1, y1, x2, y2, score, class_id]
        """
        pass
    
    @abstractmethod
    def preprocess(self, image):
        """
        Preprocess image for the specific model
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            preprocessed: Preprocessed image ready for inference
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs, image_shape):
        """
        Postprocess model outputs to get detections
        
        Args:
            outputs: Raw model outputs
            image_shape: Original image shape
            
        Returns:
            detections: Processed detections
        """
        pass


class IRImagePreprocessor:
    """Preprocessing specifically for IR illuminated images"""
    
    @staticmethod
    def enhance_ir_image(image, method='histogram'):
        """
        Enhance IR illuminated images for better detection
        
        Args:
            image: Input IR image (grayscale or RGB)
            method: Enhancement method ('histogram', 'clahe', 'gamma')
            
        Returns:
            enhanced: Enhanced image
        """
        import cv2
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if method == 'histogram':
            # Histogram equalization
            enhanced = cv2.equalizeHist(gray)
            
        elif method == 'clahe':
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
        elif method == 'gamma':
            # Gamma correction
            gamma = 1.5  # Adjust based on your IR images
            enhanced = np.power(gray / 255.0, gamma)
            enhanced = (enhanced * 255).astype(np.uint8)
            
        else:
            enhanced = gray
        
        # Convert back to 3-channel for models expecting RGB
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced
    
    @staticmethod
    def normalize_ir_range(image, min_percentile=1, max_percentile=99):
        """
        Normalize IR image dynamic range
        
        Args:
            image: Input IR image
            min_percentile: Lower percentile for normalization
            max_percentile: Upper percentile for normalization
            
        Returns:
            normalized: Normalized image
        """
        # Get percentile values for robust normalization
        min_val = np.percentile(image, min_percentile)
        max_val = np.percentile(image, max_percentile)
        
        # Normalize to 0-255 range
        normalized = np.clip((image - min_val) / (max_val - min_val), 0, 1)
        normalized = (normalized * 255).astype(np.uint8)
        
        return normalized