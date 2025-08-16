import os
import copy
import time
import argparse
import logging

import cv2

from pythonosc import udp_client
import SpoutGL
from OpenGL.GL import *
import numpy as np


from itertools import repeat
import array

from byte_tracker.byte_tracker_onnx import ByteTrackerONNX
from byte_tracker.multi_detector_tracker import create_tracker_from_args

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

UDP_IP = "127.0.0.1"
UDP_PORT = 7000

SENDER_NAME = "TouchDesigner"

# Global variables for UI controls
ui_params = {
    'track_thresh': 0.05,
    'score_th': 0.01,
    'motion_threshold': 8,
    'min_area': 200,
    'max_area': 50000,
    'min_box_area': 10
}

def on_track_thresh_change(val):
    ui_params['track_thresh'] = val / 1000.0  # Scale to 0.0-1.0
    logger.info(f"Track threshold: {ui_params['track_thresh']:.3f}")

def on_score_th_change(val):
    ui_params['score_th'] = val / 1000.0  # Scale to 0.0-1.0
    logger.info(f"Score threshold: {ui_params['score_th']:.3f}")

def on_motion_threshold_change(val):
    ui_params['motion_threshold'] = val
    logger.info(f"Motion threshold: {ui_params['motion_threshold']}")

def on_min_area_change(val):
    ui_params['min_area'] = val
    logger.info(f"Min area: {ui_params['min_area']}")

def on_max_area_change(val):
    ui_params['max_area'] = val * 100  # Scale to reasonable range
    logger.info(f"Max area: {ui_params['max_area']}")

def on_min_box_area_change(val):
    ui_params['min_box_area'] = val
    logger.info(f"Min box area: {ui_params['min_box_area']}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default='byte_tracker/model/bytetrack_s.onnx',
    )
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.7,
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default='384,640',
    )
    parser.add_argument(
        '--with_p6',
        action='store_true',
        help='Whether your model uses p6 in FPN/PAN.',
    )

    # tracking args
    parser.add_argument(
        '--track_thresh',
        type=float,
        default=0.05,
        help='tracking confidence threshold',
    )
    parser.add_argument(
        '--track_buffer',
        type=int,
        default=30,
        help='the frames for keep lost tracks',
    )
    parser.add_argument(
        '--match_thresh',
        type=float,
        default=0.8,
        help='matching threshold for tracking',
    )
    parser.add_argument(
        '--min-box-area',
        type=float,
        default=10,
        help='filter out tiny boxes',
    )
    parser.add_argument(
        '--mot20',
        dest='mot20',
        default=False,
        action='store_true',
        help='test mot20.',
    )

    # IR camera support
    parser.add_argument(
        '--ir_mode',
        action='store_true',
        help='Enable IR camera mode with image enhancement',
    )
    parser.add_argument(
        '--ir_enhancement',
        type=str,
        default='clahe',
        choices=['histogram', 'clahe', 'gamma', 'none'],
        help='IR image enhancement method',
    )
    parser.add_argument(
        '--detector_type',
        type=str,
        default='auto',
        choices=['auto', 'yolox', 'yolov5', 'yolov7', 'yolov8', 'yolo_ir', 'motion', 'adaptive_motion', 'simple_motion', 'hybrid_motion'],
        help='Detection model type',
    )
    parser.add_argument(
        '--use_legacy_tracker',
        action='store_true',
        help='Use legacy ByteTrackerONNX instead of multi-detector',
    )
    
    # Motion detection parameters
    parser.add_argument(
        '--min_area',
        type=int,
        default=200,
        help='Minimum blob area for motion detection',
    )
    parser.add_argument(
        '--max_area',
        type=int,
        default=50000,
        help='Maximum blob area for motion detection',
    )
    parser.add_argument(
        '--motion_threshold',
        type=int,
        default=8,
        help='Variance threshold for motion detection',
    )
    parser.add_argument(
        '--debug_mode',
        action='store_true',
        help='Enable debug windows and verbose logging',
    )
    parser.add_argument(
        '--ui_controls',
        action='store_true',
        help='Enable UI control sliders for real-time parameter tuning',
    )

    args = parser.parse_args()

    return args


def main():
    # 引数取得
    args = get_args()
    
    logger.info("=" * 60)
    logger.info("Starting SpoutIn ByteTrack ONNX OSC Out")
    logger.info(f"Model: {args.model}")
    logger.info(f"Detector Type: {args.detector_type}")
    logger.info(f"IR Mode: {args.ir_mode}")
    if args.ir_mode:
        logger.info(f"IR Enhancement: {args.ir_enhancement}")
    logger.info(f"Debug Mode: {args.debug_mode}")
    logger.info(f"OSC Target: {UDP_IP}:{UDP_PORT}")
    logger.info(f"Spout Sender Name: {SENDER_NAME}")
    logger.info("=" * 60)

    # cap_device = args.device
    # cap_width = args.width
    # cap_height = args.height

    logger.info(f"Initializing OSC client at {UDP_IP}:{UDP_PORT}")
    client = udp_client.SimpleUDPClient(UDP_IP, UDP_PORT)

    # ByteTrackerインスタンス生成
    if args.use_legacy_tracker:
        logger.info("Initializing legacy ByteTracker with ONNX...")
        byte_tracker = ByteTrackerONNX(args)
    else:
        logger.info("Initializing multi-detector ByteTracker...")
        byte_tracker = create_tracker_from_args(
            args=args,
            detector_type=args.detector_type,
            is_ir_mode=args.ir_mode,
            ir_enhancement=args.ir_enhancement if args.ir_enhancement != 'none' else None,
            debug_mode=args.debug_mode
        )
    logger.info("ByteTracker initialized successfully")
    
    # Initialize UI parameters from command line args
    ui_params['track_thresh'] = args.track_thresh
    ui_params['score_th'] = args.score_th
    ui_params['motion_threshold'] = args.motion_threshold
    ui_params['min_area'] = args.min_area
    ui_params['max_area'] = args.max_area
    ui_params['min_box_area'] = getattr(args, 'min_box_area', 10)

    # # カメラ準備
    # cap = cv2.VideoCapture(cap_device)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    frame_id = 1

    # Create OpenCV windows
    logger.info("Creating display window...")
    cv2.namedWindow('Spout In ByteTrack ONNX OSC Out', cv2.WINDOW_NORMAL)
    
    # Create debug windows if debug mode is enabled
    if args.debug_mode:
        logger.info("Debug mode enabled - creating debug windows...")
        cv2.namedWindow('Debug: Motion Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Debug: Detections', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Debug: Tracking', cv2.WINDOW_NORMAL)
        
        # Position debug windows side by side (will be sized properly once we get first frame)
        cv2.moveWindow('Debug: Motion Mask', 680, 50)     # Top right
        cv2.moveWindow('Debug: Detections', 680, 420)     # Bottom right
        cv2.moveWindow('Debug: Tracking', 1360, 50)       # Far right
    
    # Show a black placeholder image initially
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, f"Waiting for Spout sender: {SENDER_NAME}", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Spout In ByteTrack ONNX OSC Out', placeholder)
    cv2.waitKey(1)
    
    logger.info("Initializing Spout receiver...")
    with SpoutGL.SpoutReceiver() as receiver:
        receiver.setReceiverName(SENDER_NAME)
        logger.info(f"Spout receiver created, waiting for sender: '{SENDER_NAME}'...")

        buffer = None
        first_frame = True
        no_input_logged = False

        while True:
            start_time = time.time()

            # # フレーム読み出し
            # ret, frame = cap.read()
            # if not ret:
            #     break

            # Did we get an image?
            result = receiver.receiveImage(buffer, GL_RGB, False, 0)

            # Resize if sender chages size
            if receiver.isUpdated():
                width = receiver.getSenderWidth()
                height = receiver.getSenderHeight()
                buffer = array.array('B', repeat(0, width * height * 3))
                logger.info(f"Spout sender detected! Size: {width}x{height}")

            if buffer and result and not SpoutGL.helpers.isBufferEmpty(buffer):
                if first_frame:
                    logger.info(f"First frame received! Processing at {width}x{height}")
                    first_frame = False
                    no_input_logged = False
                    
                    # Resize and reposition debug windows to fixed size that fits on screen
                    if args.debug_mode:
                        # Use fixed debug window size (400x300) that will fit on most screens
                        debug_width = 400
                        debug_height = 300
                        
                        # Resize all debug windows to fixed size
                        cv2.resizeWindow('Debug: Motion Mask', debug_width, debug_height)
                        cv2.resizeWindow('Debug: Detections', debug_width, debug_height)
                        cv2.resizeWindow('Debug: Tracking', debug_width, debug_height)
                        
                        # Position debug windows in a simple grid (main window can overlap)
                        cv2.moveWindow('Debug: Motion Mask', 50, 50)          # Top left
                        cv2.moveWindow('Debug: Detections', 50, 370)         # Bottom left  
                        cv2.moveWindow('Debug: Tracking', 470, 50)           # Top right
                    
                    # Create UI control window if enabled (only once)
                    if args.ui_controls and not hasattr(main, '_ui_created'):
                        logger.info("Creating UI control window...")
                        cv2.namedWindow('Motion Detection Controls', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Motion Detection Controls', 400, 500)
                        cv2.moveWindow('Motion Detection Controls', 900, 50)
                        
                        # Create trackbars for real-time parameter adjustment
                        cv2.createTrackbar('Track Thresh x1000', 'Motion Detection Controls', 
                                          int(ui_params['track_thresh'] * 1000), 1000, on_track_thresh_change)
                        cv2.createTrackbar('Score Thresh x1000', 'Motion Detection Controls', 
                                          int(ui_params['score_th'] * 1000), 1000, on_score_th_change)
                        cv2.createTrackbar('Motion Threshold', 'Motion Detection Controls', 
                                          ui_params['motion_threshold'], 50, on_motion_threshold_change)
                        cv2.createTrackbar('Min Area', 'Motion Detection Controls', 
                                          ui_params['min_area'], 2000, on_min_area_change)
                        cv2.createTrackbar('Max Area /100', 'Motion Detection Controls', 
                                          ui_params['max_area'] // 100, 1000, on_max_area_change)
                        cv2.createTrackbar('Min Box Area', 'Motion Detection Controls', 
                                          ui_params['min_box_area'], 1000, on_min_box_area_change)
                        
                        # Create a black image with instructions
                        control_img = np.zeros((500, 400, 3), dtype=np.uint8)
                        cv2.putText(control_img, "Motion Detection Controls", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(control_img, "Adjust sliders in real-time", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        cv2.putText(control_img, "Track Thresh: Detection confidence", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                        cv2.putText(control_img, "Score Thresh: Minimum score", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                        cv2.putText(control_img, "Motion Thresh: Motion sensitivity", (10, 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                        cv2.putText(control_img, "Min/Max Area: Blob size limits", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                        cv2.imshow('Motion Detection Controls', control_img)
                        main._ui_created = True
                # print("Got bytes", bytes(buffer[0:64]), "...")

                frame = np.array(buffer)
                frame = np.reshape(frame, (height, width, 3))
                
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                debug_image = copy.deepcopy(frame)

                # Byte Tracker推論 with debug info
                if args.debug_mode and hasattr(byte_tracker, 'detector') and hasattr(byte_tracker.detector, 'detect'):
                    # Get raw detections from motion detector for debug
                    raw_detections = byte_tracker.detector.detect(frame)
                    logger.debug(f"Frame {frame_id}: Raw motion detections: {len(raw_detections)}")
                    
                    # Create debug images showing motion detection stages (only every 3rd frame to avoid blocking)
                    if frame_id % 3 == 0:
                        if hasattr(byte_tracker.detector, 'get_debug_image'):
                            motion_debug = byte_tracker.detector.get_debug_image(frame)
                            cv2.imshow('Debug: Motion Mask', motion_debug)
                        
                        # Create detection visualization
                        detection_debug = frame.copy()
                        for i, det in enumerate(raw_detections):
                            x1, y1, x2, y2, score, class_id = det
                            cv2.rectangle(detection_debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(detection_debug, f"{score:.3f}", (int(x1), int(y1-5)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(detection_debug, f"Raw Detections: {len(raw_detections)}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Debug: Detections', detection_debug)

                # Update tracker parameters from UI if controls are enabled
                if args.ui_controls:
                    # Update ByteTracker thresholds
                    byte_tracker.args.track_thresh = ui_params['track_thresh']
                    byte_tracker.args.score_th = ui_params['score_th']
                    byte_tracker.args.min_box_area = ui_params['min_box_area']
                    byte_tracker.tracker.det_thresh = ui_params['track_thresh']  # Update internal threshold
                    
                    # Update motion detector parameters if available
                    if hasattr(byte_tracker, 'detector') and hasattr(byte_tracker.detector, 'min_area'):
                        byte_tracker.detector.min_area = ui_params['min_area']
                        byte_tracker.detector.max_area = ui_params['max_area']
                        # Update background subtractor variance threshold if it's a motion detector
                        if hasattr(byte_tracker.detector, 'bg_subtractor'):
                            byte_tracker.detector.bg_subtractor.setVarThreshold(ui_params['motion_threshold'])

                # Run tracker inference
                _, bboxes, ids, scores = byte_tracker.inference(frame)

                elapsed_time = time.time() - start_time

                # Enhanced logging for debug mode
                if args.debug_mode:
                    logger.info(f"Frame {frame_id}: Raw->Tracker: {len(raw_detections) if 'raw_detections' in locals() else 'N/A'}->{len(bboxes)} objects, inference: {elapsed_time*1000:.2f}ms")
                elif frame_id % 30 == 0:  # Log every 30 frames in normal mode
                    logger.info(f"Frame {frame_id}: Detected {len(bboxes)} objects, inference time: {elapsed_time*1000:.2f}ms")
                
                # Debug logging for tracking pipeline
                if args.debug_mode and frame_id % 10 == 0:  # Log every 10 frames in debug
                    if 'raw_detections' in locals():
                        logger.debug(f"  Raw detections: {len(raw_detections)}")
                        for i, det in enumerate(raw_detections[:3]):  # Show first 3
                            logger.debug(f"    Det {i}: bbox=({det[0]:.0f},{det[1]:.0f},{det[2]:.0f},{det[3]:.0f}) score={det[4]:.3f}")
                    logger.debug(f"  Final tracks: {len(bboxes)}")
                    if len(bboxes) > 0:
                        for i, (bbox, track_id, score) in enumerate(zip(bboxes[:3], ids[:3], scores[:3])):
                            logger.debug(f"    Track {i}: id={track_id} bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}) score={score:.3f}")

                # 検出情報描画
                debug_image = draw_tracking_info(
                    debug_image,
                    bboxes,
                    ids,
                    scores,
                    frame_id,
                    elapsed_time,
                )
                
                # Create tracking debug window (only every 3rd frame to avoid blocking)
                if args.debug_mode and frame_id % 3 == 0:
                    tracking_debug = frame.copy()
                    for i, (bbox, track_id, score) in enumerate(zip(bboxes, ids, scores)):
                        x1, y1 = int(bbox[0]), int(bbox[1])
                        x2, y2 = x1 + int(bbox[2]), y1 + int(bbox[3])
                        cv2.rectangle(tracking_debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(tracking_debug, f"ID:{track_id} {score:.3f}", (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(tracking_debug, f"Final Tracks: {len(bboxes)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Debug: Tracking', tracking_debug)

                # キー処理(ESC：終了)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    logger.info("ESC pressed, shutting down...")
                    break

                # 画面反映
                cv2.imshow('Spout In ByteTrack ONNX OSC Out', debug_image)

                frame_id += 1
        
                bboxeslist = []
                for i in range(len(bboxes)):
                    bboxeslist.extend(bboxes[i])
                    bboxeslist.append(ids[i])
                    bboxeslist.append(scores[i])
                    
                if len(bboxeslist) > 0:
                    client.send_message('/bboxes', bboxeslist)
                    if frame_id % 30 == 0:  # Log OSC messages periodically
                        logger.debug(f"Sent OSC message with {len(bboxes)} detections")

                rtime = (time.time() - start_time)*1000
                if frame_id % 30 == 0:
                    logger.info(f"Total frame processing time: {rtime:.2f} ms")

                receiver.waitFrameSync(SENDER_NAME, 10000)
            else:
                # No input received
                if not no_input_logged:
                    logger.warning(f"No Spout input received from '{SENDER_NAME}'. Please ensure TouchDesigner or another Spout sender is running.")
                    logger.info("Waiting for Spout sender... (Press ESC in the window to exit)")
                    no_input_logged = True
                
                # Still check for ESC key
                key = cv2.waitKey(100)
                if key == 27:  # ESC
                    logger.info("ESC pressed, shutting down...")
                    break


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_tracking_info(
    image,
    tlwhs,
    ids,
    scores,
    frame_id=0,
    elapsed_time=0.,
):
    text_scale = 1.5
    text_thickness = 2
    line_thickness = 2

    # フレーム数、処理時間、推論時間
    text = 'frame: %d ' % (frame_id)
    text += 'elapsed time: %.0fms ' % (elapsed_time * 1000)
    text += 'num: %d' % (len(tlwhs))
    cv2.putText(
        image,
        text,
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        thickness=text_thickness,
    )

    for index, tlwh in enumerate(tlwhs):
        x1, y1 = int(tlwh[0]), int(tlwh[1])
        x2, y2 = x1 + int(tlwh[2]), y1 + int(tlwh[3])

        # バウンディングボックス
        color = get_id_color(ids[index])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # ID、スコア
        # text = str(ids[index]) + ':%.2f' % (scores[index])
        text = str(ids[index])
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 0, 0), text_thickness + 3)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (255, 255, 255), text_thickness)
    return image


if __name__ == '__main__':
    main()
