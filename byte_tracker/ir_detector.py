#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IR-specific detector for models trained on thermal/infrared data
"""

import numpy as np
import onnxruntime
import logging
import cv2
from .detector_base import DetectorBase

logger = logging.getLogger(__name__)


class IRDetector(DetectorBase):
    """
    Detector specifically for IR-trained models
    These models are already trained on IR/thermal data so don't need RGB preprocessing
    """
    
    def __init__(self, model_path, input_shape=None, providers=None,
                 score_thresh=0.25, nms_thresh=0.45, model_type='yolov5'):
        super().__init__(model_path, input_shape, providers)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.model_type = model_type.lower()
        
        # Load ONNX model
        logger.info(f"Loading IR-trained model from: {model_path}")
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=self.providers
        )
        
        # Auto-detect input shape
        if self.input_shape is None:
            model_inputs = self.session.get_inputs()
            input_shape_from_model = model_inputs[0].shape
            if len(input_shape_from_model) >= 4:
                self.input_shape = (input_shape_from_model[2], input_shape_from_model[3])
                logger.info(f"Auto-detected input shape: {self.input_shape}")
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"IR detector initialized:")
        logger.info(f"  Model type: {self.model_type}")
        logger.info(f"  Input shape: {self.input_shape}")
        logger.info(f"  Input name: {self.input_name}")
        logger.info(f"  Output names: {self.output_names}")
    
    def detect(self, image):
        """
        Perform detection on IR image
        
        Args:
            image: Input IR image (grayscale or RGB, will be converted appropriately)
            
        Returns:
            detections: Array of [x1, y1, x2, y2, score, class_id]
        """
        # Preprocess image
        preprocessed = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
        
        # Postprocess based on model type
        detections = self.postprocess(outputs, image.shape)
        
        return detections
    
    def preprocess(self, image):
        """
        Preprocess IR image for IR-trained model
        
        Args:
            image: Input IR image
            
        Returns:
            preprocessed: Preprocessed image ready for inference
        """
        # Convert to grayscale if needed (IR images are typically grayscale)
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Convert RGB to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image[:, :, 0]  # Take first channel
        else:
            gray = image
        
        # Convert back to 3-channel for models expecting RGB input
        if len(gray.shape) == 2:
            image_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            image_3ch = gray
        
        # Resize to model input size
        h, w = self.input_shape
        resized = cv2.resize(image_3ch, (w, h))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess(self, outputs, image_shape):
        """
        Postprocess model outputs
        
        Args:
            outputs: Raw model outputs
            image_shape: Original image shape
            
        Returns:
            detections: Processed detections [x1, y1, x2, y2, score, class_id]
        """
        if self.model_type in ['yolov5', 'yolov8']:
            return self._postprocess_yolo(outputs[0], image_shape)
        elif self.model_type == 'yolox':
            return self._postprocess_yolox(outputs[0], image_shape)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using YOLOv5 postprocessing")
            return self._postprocess_yolo(outputs[0], image_shape)
    
    def _postprocess_yolo(self, output, image_shape):
        """Postprocess YOLOv5/v8 outputs"""
        # YOLOv5/v8 output format: [batch, num_detections, 85] where 85 = 4 (bbox) + 1 (conf) + 80 (classes)
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Filter by confidence
        conf_mask = output[:, 4] > self.score_thresh
        output = output[conf_mask]
        
        if len(output) == 0:
            return np.empty((0, 6))
        
        # Extract boxes, confidence, and class scores
        boxes = output[:, :4]  # x_center, y_center, width, height
        conf = output[:, 4]
        class_scores = output[:, 5:]
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        max_class_scores = np.max(class_scores, axis=1)
        
        # Combine confidence and class score
        scores = conf * max_class_scores
        
        # Filter by final score
        score_mask = scores > self.score_thresh
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        class_ids = class_ids[score_mask]
        
        if len(boxes) == 0:
            return np.empty((0, 6))
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale coordinates to original image size
        h_scale = image_shape[0] / self.input_shape[0]
        w_scale = image_shape[1] / self.input_shape[1]
        
        x1 *= w_scale
        x2 *= w_scale
        y1 *= h_scale
        y2 *= h_scale
        
        # Apply NMS
        keep = self._nms(np.column_stack([x1, y1, x2, y2]), scores, self.nms_thresh)
        
        # Return format: [x1, y1, x2, y2, score, class_id]
        detections = np.column_stack([
            x1[keep], y1[keep], x2[keep], y2[keep],
            scores[keep], class_ids[keep]
        ])
        
        return detections
    
    def _postprocess_yolox(self, output, image_shape):
        """Postprocess YOLOX outputs"""
        # YOLOX typically has different output format
        # This is a simplified version - adjust based on actual YOLOX IR model
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # YOLOX format: [x1, y1, x2, y2, conf, class_conf, class_id]
        boxes = output[:, :4]
        conf = output[:, 4]
        
        # Filter by confidence
        conf_mask = conf > self.score_thresh
        boxes = boxes[conf_mask]
        conf = conf[conf_mask]
        
        if len(boxes) == 0:
            return np.empty((0, 6))
        
        # Scale to original image size
        h_scale = image_shape[0] / self.input_shape[0]
        w_scale = image_shape[1] / self.input_shape[1]
        
        boxes[:, [0, 2]] *= w_scale  # x coordinates
        boxes[:, [1, 3]] *= h_scale  # y coordinates
        
        # Apply NMS
        keep = self._nms(boxes, conf, self.nms_thresh)
        
        # Assume person class (0) for IR person detection
        class_ids = np.zeros(len(keep), dtype=int)
        
        detections = np.column_stack([
            boxes[keep], conf[keep], class_ids
        ])
        
        return detections
    
    def _nms(self, boxes, scores, threshold):
        """Non-maximum suppression"""
        if len(boxes) == 0:
            return []
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.score_thresh,
            threshold
        )
        
        if len(indices) > 0:
            return indices.flatten()
        return []


class FlirThermalDetector(IRDetector):
    """
    Specialized detector for FLIR thermal dataset trained models
    """
    
    def __init__(self, model_path, **kwargs):
        # FLIR dataset classes: person, bicycle, car, dog
        self.flir_classes = ['person', 'bicycle', 'car', 'dog']
        super().__init__(model_path, **kwargs)
        
        logger.info(f"FLIR thermal detector initialized with classes: {self.flir_classes}")
    
    def detect(self, image):
        """Enhanced detection for FLIR thermal images"""
        # FLIR images might need different preprocessing
        
        # Apply thermal-specific enhancement if needed
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Single channel thermal image
            enhanced = self._enhance_thermal_image(image)
        else:
            enhanced = image
        
        return super().detect(enhanced)
    
    def _enhance_thermal_image(self, image):
        """
        Apply thermal-specific enhancements
        
        Args:
            image: Single channel thermal image
            
        Returns:
            enhanced: Enhanced thermal image
        """
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Take first channel
        
        # Normalize thermal range (important for thermal images)
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        
        if max_val > min_val:
            normalized = np.clip((image - min_val) / (max_val - min_val), 0, 1)
            normalized = (normalized * 255).astype(np.uint8)
        else:
            normalized = image
        
        # Apply mild contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        return enhanced