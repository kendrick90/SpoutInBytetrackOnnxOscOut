#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-detector ByteTracker supporting different detection models
"""

import logging
import numpy as np
from .tracker.byte_tracker import BYTETracker
from .yolox_detector import YOLOXDetector, YOLODetector
from .yolo_ir_detector import YOLOIRDetector, FLIRYOLODetector

logger = logging.getLogger(__name__)


class MultiDetectorByteTracker:
    """ByteTracker with pluggable detection backends"""
    
    def __init__(self, args, detector_type='yolox', is_ir_mode=False, 
                 ir_enhancement='clahe', frame_rate=30, use_ir_trained_model=False):
        self.args = args
        self.detector_type = detector_type
        self.is_ir_mode = is_ir_mode
        self.use_ir_trained_model = use_ir_trained_model
        
        # Initialize detector based on type
        if detector_type == 'yolo_ir':
            # Use YOLO-IR detector for IR-trained models
            if 'flir' in args.model.lower():
                logger.info("Using FLIR YOLO detector")
                self.detector = FLIRYOLODetector(
                    model_path=args.model,
                    score_thresh=args.score_th,
                    nms_thresh=args.nms_th
                )
            else:
                logger.info("Using YOLO-IR detector")
                self.detector = YOLOIRDetector(
                    model_path=args.model,
                    score_thresh=args.score_th,
                    nms_thresh=args.nms_th
                )
        elif detector_type == 'yolox':
            self.detector = YOLOXDetector(
                model_path=args.model,
                score_thresh=args.score_th,
                nms_thresh=args.nms_th,
                is_ir_mode=is_ir_mode,
                ir_enhancement=ir_enhancement
            )
        elif detector_type in ['yolov5', 'yolov7', 'yolov8']:
            yolo_version = detector_type[-2:]  # Extract version number
            self.detector = YOLODetector(
                model_path=args.model,
                score_thresh=args.score_th,
                nms_thresh=args.nms_th,
                is_ir_mode=is_ir_mode,
                ir_enhancement=ir_enhancement,
                yolo_version=yolo_version
            )
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
        
        # Initialize ByteTracker
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        
        logger.info(f"MultiDetectorByteTracker initialized:")
        logger.info(f"  Detector: {detector_type}")
        logger.info(f"  IR Mode: {is_ir_mode}")
        logger.info(f"  Enhancement: {ir_enhancement}")
    
    def inference(self, image):
        """
        Run detection and tracking on image
        
        Args:
            image: Input image (RGB or IR)
            
        Returns:
            output_image: Image with detection visualization
            bboxes: List of bounding boxes
            ids: List of track IDs
            scores: List of detection scores
        """
        # Run detection
        detections = self.detector.detect(image)
        
        # Convert detections to tracker format
        if len(detections) > 0:
            # Format: [x1, y1, x2, y2, score]
            det_boxes = detections[:, :4]
            det_scores = detections[:, 4]
            det_classes = detections[:, 5] if detections.shape[1] > 5 else None
            
            # Filter for person class (class 0) if available
            if det_classes is not None:
                person_mask = det_classes == 0  # Person class in COCO
                det_boxes = det_boxes[person_mask]
                det_scores = det_scores[person_mask]
            
            # Combine boxes and scores for tracker format
            if len(det_boxes) > 0:
                output_results = np.column_stack([det_boxes, det_scores])
            else:
                output_results = np.empty((0, 5))
        else:
            output_results = np.empty((0, 5))
        
        # Update tracker
        online_targets = self.tracker.update(
            output_results=output_results,
            img_info=[image.shape[0], image.shape[1]],
            img_size=[image.shape[0], image.shape[1]]
        )
        
        # Extract tracking results
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        for track in online_targets:
            tlwh = track.tlwh
            track_id = track.track_id
            score = track.score
            
            if tlwh[2] * tlwh[3] > self.args.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_scores.append(score)
        
        return image, online_tlwhs, online_ids, online_scores
    
    def set_ir_mode(self, is_ir_mode, enhancement=None):
        """
        Switch between RGB and IR mode at runtime
        
        Args:
            is_ir_mode: Whether to enable IR preprocessing
            enhancement: IR enhancement method
        """
        self.is_ir_mode = is_ir_mode
        self.detector.is_ir_mode = is_ir_mode
        
        if enhancement:
            self.detector.ir_enhancement = enhancement
        
        logger.info(f"Switched to {'IR' if is_ir_mode else 'RGB'} mode")
        if enhancement:
            logger.info(f"IR enhancement: {enhancement}")


def create_tracker_from_args(args, detector_type='auto', is_ir_mode=False, 
                           ir_enhancement='clahe', use_ir_trained_model=False):
    """
    Factory function to create tracker with automatic detector type detection
    
    Args:
        args: Command line arguments
        detector_type: Detector type ('auto', 'yolox', 'yolov5', etc.)
        is_ir_mode: Enable IR preprocessing for RGB models
        ir_enhancement: IR enhancement method
        use_ir_trained_model: Use IR-specific detector for IR-trained models
        
    Returns:
        MultiDetectorByteTracker instance
    """
    if detector_type == 'auto':
        # Try to detect model type from filename
        model_name = args.model.lower()
        if any(keyword in model_name for keyword in ['flir', 'ir_', '_ir', 'thermal', 'infrared']):
            detector_type = 'yolo_ir'
            logger.info("Auto-detected IR-trained model")
        elif 'yolox' in model_name or 'bytetrack' in model_name:
            detector_type = 'yolox'
        elif 'yolov8' in model_name:
            detector_type = 'yolov8'
        elif 'yolov7' in model_name:
            detector_type = 'yolov7'
        elif 'yolov5' in model_name:
            detector_type = 'yolov5'
        else:
            # Default to YOLOX for compatibility
            detector_type = 'yolox'
            logger.warning(f"Could not detect model type from {args.model}, defaulting to YOLOX")
    
    # Auto-detect if this is an IR-trained model
    model_name = args.model.lower()
    if not use_ir_trained_model:
        if any(keyword in model_name for keyword in ['flir', 'thermal', 'infrared', 'ir_trained', '_ir_']):
            use_ir_trained_model = True
            logger.info("Auto-detected IR-trained model, using IR detector")
    
    logger.info(f"Creating tracker with detector type: {detector_type}")
    if use_ir_trained_model:
        logger.info("Using IR-trained model detector (no RGB preprocessing)")
    elif is_ir_mode:
        logger.info("Using RGB model with IR preprocessing")
    
    return MultiDetectorByteTracker(
        args=args,
        detector_type=detector_type,
        is_ir_mode=is_ir_mode,
        ir_enhancement=ir_enhancement,
        use_ir_trained_model=use_ir_trained_model
    )