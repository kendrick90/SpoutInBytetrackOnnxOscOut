#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOX detector implementation with IR support
"""

import numpy as np
import onnxruntime
import logging
from .detector_base import DetectorBase, IRImagePreprocessor
from .utils.yolox_utils import pre_process as preproc, multiclass_nms

logger = logging.getLogger(__name__)


class YOLOXDetector(DetectorBase):
    """YOLOX detector with optional IR image preprocessing"""
    
    def __init__(self, model_path, input_shape=None, providers=None, 
                 score_thresh=0.3, nms_thresh=0.45, is_ir_mode=False, 
                 ir_enhancement='clahe'):
        super().__init__(model_path, input_shape, providers)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.is_ir_mode = is_ir_mode
        self.ir_enhancement = ir_enhancement
        
        # RGB normalization parameters (YOLOX defaults)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # Load ONNX model
        logger.info(f"Loading YOLOX model from: {model_path}")
        self.session = onnxruntime.InferenceSession(
            model_path, 
            providers=self.providers
        )
        
        # Auto-detect input shape if not provided
        if self.input_shape is None:
            model_inputs = self.session.get_inputs()
            input_shape_from_model = model_inputs[0].shape
            if input_shape_from_model[2] > 0 and input_shape_from_model[3] > 0:
                self.input_shape = (input_shape_from_model[2], input_shape_from_model[3])
                logger.info(f"Auto-detected input shape: {self.input_shape}")
        
        logger.info(f"YOLOX initialized - IR mode: {is_ir_mode}, Shape: {self.input_shape}")
        
    def detect(self, image):
        """
        Perform object detection
        
        Args:
            image: Input image (numpy array, RGB or IR)
            
        Returns:
            detections: Array of [x1, y1, x2, y2, score, class_id]
        """
        # Apply IR preprocessing if needed
        if self.is_ir_mode:
            image = self._preprocess_ir(image)
        
        # Standard preprocessing
        preprocessed, ratio = self.preprocess(image)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: preprocessed[None, :, :, :]})[0]
        
        # Postprocess
        detections = self.postprocess(outputs, image.shape, ratio)
        
        return detections
    
    def preprocess(self, image):
        """
        Standard YOLOX preprocessing
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            preprocessed: Preprocessed image
            ratio: Scaling ratio
        """
        preprocessed, ratio = preproc(image, self.input_shape, self.rgb_means, self.std)
        return preprocessed, ratio
    
    def postprocess(self, outputs, image_shape, ratio):
        """
        Postprocess YOLOX outputs
        
        Args:
            outputs: Model outputs
            image_shape: Original image shape
            ratio: Preprocessing ratio
            
        Returns:
            detections: Array of detections [x1, y1, x2, y2, score, class_id]
        """
        predictions = outputs[0]
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        
        dets = multiclass_nms(
            boxes_xyxy, 
            scores, 
            nms_thr=self.nms_thresh, 
            score_thr=self.score_thresh
        )
        
        if dets is not None:
            return dets[:, :6]  # x1, y1, x2, y2, score, class_id
        return np.empty((0, 6))
    
    def _preprocess_ir(self, image):
        """
        Apply IR-specific preprocessing
        
        Args:
            image: Input IR image
            
        Returns:
            enhanced: Enhanced IR image
        """
        logger.debug(f"Applying IR enhancement: {self.ir_enhancement}")
        
        # First normalize the IR range
        image = IRImagePreprocessor.normalize_ir_range(image)
        
        # Then apply enhancement
        enhanced = IRImagePreprocessor.enhance_ir_image(image, self.ir_enhancement)
        
        return enhanced


class YOLODetector(DetectorBase):
    """
    Generic YOLO detector (v5, v7, v8) with IR support
    Can work with different YOLO variants
    """
    
    def __init__(self, model_path, input_shape=None, providers=None,
                 score_thresh=0.25, nms_thresh=0.45, is_ir_mode=False,
                 ir_enhancement='clahe', yolo_version='v7'):
        super().__init__(model_path, input_shape, providers)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.is_ir_mode = is_ir_mode
        self.ir_enhancement = ir_enhancement
        self.yolo_version = yolo_version
        
        # Load ONNX model
        logger.info(f"Loading YOLO{yolo_version} model from: {model_path}")
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
        
        logger.info(f"YOLO{yolo_version} initialized - IR mode: {is_ir_mode}")
    
    def detect(self, image):
        """Perform detection with IR preprocessing if needed"""
        if self.is_ir_mode:
            image = IRImagePreprocessor.enhance_ir_image(image, self.ir_enhancement)
        
        # Preprocess
        preprocessed = self.preprocess(image)
        
        # Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: preprocessed})
        
        # Postprocess based on YOLO version
        detections = self.postprocess(outputs[0], image.shape)
        
        return detections
    
    def preprocess(self, image):
        """YOLO preprocessing (letterbox, normalization)"""
        import cv2
        
        # Resize with letterbox
        h, w = self.input_shape
        image_resized = cv2.resize(image, (w, h))
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # HWC to CHW
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        
        # Add batch dimension
        image_batched = np.expand_dims(image_transposed, axis=0).astype(np.float32)
        
        return image_batched
    
    def postprocess(self, outputs, image_shape):
        """
        Postprocess YOLO outputs
        Note: This is simplified and may need adjustment based on specific YOLO version
        """
        # This is a simplified version - actual implementation depends on YOLO version
        # and output format of your specific model
        
        # Filter by confidence
        predictions = outputs[outputs[..., 4] > self.score_thresh]
        
        if len(predictions) == 0:
            return np.empty((0, 6))
        
        # Format: [x1, y1, x2, y2, score, class_id]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5:].argmax(axis=1)
        
        # Apply NMS
        keep = self._nms(boxes, scores, self.nms_thresh)
        
        detections = np.column_stack([
            boxes[keep],
            scores[keep],
            class_ids[keep]
        ])
        
        return detections
    
    def _nms(self, boxes, scores, threshold):
        """Simple NMS implementation"""
        # This is a placeholder - you might want to use cv2.dnn.NMSBoxes
        # or a more sophisticated implementation
        import cv2
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.score_thresh,
            threshold
        )
        
        if len(indices) > 0:
            return indices.flatten()
        return []