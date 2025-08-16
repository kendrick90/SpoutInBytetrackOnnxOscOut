#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean YOLO-IR detector for models trained specifically on IR/infrared data
No preprocessing - these models are already trained for IR
"""

import numpy as np
import onnxruntime
import logging
import cv2

logger = logging.getLogger(__name__)


class YOLOIRDetector:
    """
    Detector for YOLO models trained specifically on infrared data
    Examples: YOLO-IR-Free, YOLOv5 trained on FLIR dataset, etc.
    """
    
    def __init__(self, model_path, score_thresh=0.25, nms_thresh=0.45, 
                 input_size=640, providers=None):
        self.model_path = model_path
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.input_height = input_size  # Default values
        self.input_width = input_size   # Will be overridden by auto-detection
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Load ONNX model
        logger.info(f"Loading YOLO-IR model from: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Auto-detect input shape
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 4 and input_shape[2] > 0 and input_shape[3] > 0:
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            # For compatibility, keep input_size as height (legacy)
            self.input_size = self.input_height
        
        actual_providers = self.session.get_providers()
        logger.info(f"YOLO-IR detector initialized:")
        logger.info(f"  Input size: {self.input_height}x{self.input_width}")
        logger.info(f"  Providers: {actual_providers}")
        logger.info(f"  Input name: {self.input_name}")
        logger.info(f"  Output names: {self.output_names}")
    
    def detect(self, image):
        """
        Detect objects in IR image using IR-trained YOLO model
        
        Args:
            image: Input IR image (grayscale or RGB)
            
        Returns:
            detections: Array of [x1, y1, x2, y2, score, class_id]
        """
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self._postprocess(outputs[0], image.shape)
        
        return detections
    
    def _preprocess(self, image):
        """
        Preprocess image for YOLO-IR model
        No enhancement - model is already trained for IR
        """
        # Handle grayscale IR images
        if len(image.shape) == 2:
            # Single channel grayscale - convert to 3-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel in 3D array
            image = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        input_tensor = np.expand_dims(transposed, axis=0)
        
        return input_tensor
    
    def _postprocess(self, output, original_shape):
        """
        Postprocess YOLO output to get detections
        """
        # Handle different YOLO output formats
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # YOLOv5/v8 format: [num_detections, 85] where 85 = 4 (bbox) + 1 (conf) + 80 (classes)
        # Extract boxes, confidence, and class scores
        boxes = output[:, :4]  # x_center, y_center, width, height
        confidence = output[:, 4]
        class_scores = output[:, 5:]
        
        # Filter by confidence threshold
        conf_mask = confidence > self.score_thresh
        boxes = boxes[conf_mask]
        confidence = confidence[conf_mask]
        class_scores = class_scores[conf_mask]
        
        if len(boxes) == 0:
            return np.empty((0, 6))
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        max_class_scores = np.max(class_scores, axis=1)
        
        # Combine confidence and class score
        final_scores = confidence * max_class_scores
        
        # Filter by final score
        score_mask = final_scores > self.score_thresh
        boxes = boxes[score_mask]
        final_scores = final_scores[score_mask]
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
        h_scale = original_shape[0] / self.input_height
        w_scale = original_shape[1] / self.input_width
        
        x1 *= w_scale
        x2 *= w_scale
        y1 *= h_scale
        y2 *= h_scale
        
        # Prepare boxes for NMS
        boxes_for_nms = np.column_stack([x1, y1, x2, y2])
        
        # Apply NMS
        keep_indices = self._nms(boxes_for_nms, final_scores)
        
        # Return final detections: [x1, y1, x2, y2, score, class_id]
        if len(keep_indices) > 0:
            detections = np.column_stack([
                boxes_for_nms[keep_indices],
                final_scores[keep_indices],
                class_ids[keep_indices]
            ])
            return detections
        
        return np.empty((0, 6))
    
    def _nms(self, boxes, scores):
        """Non-maximum suppression"""
        if len(boxes) == 0:
            return []
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.score_thresh,
            self.nms_thresh
        )
        
        if len(indices) > 0:
            return indices.flatten()
        return []


class FLIRYOLODetector(YOLOIRDetector):
    """
    Specialized detector for YOLO models trained on FLIR thermal dataset
    """
    
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        
        # FLIR dataset classes: person, bicycle, car, dog
        self.flir_classes = ['person', 'bicycle', 'car', 'dog']
        logger.info(f"FLIR YOLO detector loaded with classes: {self.flir_classes}")
    
    def get_class_name(self, class_id):
        """Get class name for FLIR dataset"""
        if 0 <= class_id < len(self.flir_classes):
            return self.flir_classes[class_id]
        return f"class_{class_id}"


class YOLOIRFreeDetector(YOLOIRDetector):
    """
    Detector for YOLO-IR-Free models (improved IR vehicle detection)
    """
    
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        logger.info("YOLO-IR-Free detector initialized")
    
    def _postprocess(self, output, original_shape):
        """
        Custom postprocessing for YOLO-IR-Free if needed
        Falls back to standard YOLO postprocessing
        """
        # YOLO-IR-Free might have custom output format
        # For now, use standard YOLO postprocessing
        return super()._postprocess(output, original_shape)