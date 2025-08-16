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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

UDP_IP = "127.0.0.1"
UDP_PORT = 7000

SENDER_NAME = "TouchDesigner"


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
        default=0.1,
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
        default=0.5,
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

    args = parser.parse_args()

    return args


def main():
    # 引数取得
    args = get_args()
    
    logger.info("=" * 60)
    logger.info("Starting SpoutIn ByteTrack ONNX OSC Out")
    logger.info(f"Model: {args.model}")
    logger.info(f"OSC Target: {UDP_IP}:{UDP_PORT}")
    logger.info(f"Spout Sender Name: {SENDER_NAME}")
    logger.info("=" * 60)

    # cap_device = args.device
    # cap_width = args.width
    # cap_height = args.height

    logger.info(f"Initializing OSC client at {UDP_IP}:{UDP_PORT}")
    client = udp_client.SimpleUDPClient(UDP_IP, UDP_PORT)

    # ByteTrackerインスタンス生成
    logger.info("Initializing ByteTracker with ONNX...")
    byte_tracker = ByteTrackerONNX(args)
    logger.info("ByteTracker initialized successfully")

    # # カメラ準備
    # cap = cv2.VideoCapture(cap_device)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    frame_id = 1

    # Create OpenCV window early
    logger.info("Creating display window...")
    cv2.namedWindow('Spout In ByteTrack ONNX OSC Out', cv2.WINDOW_NORMAL)
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
                # print("Got bytes", bytes(buffer[0:64]), "...")

                frame = np.array(buffer)
                frame = np.reshape(frame, (height, width, 3))

                debug_image = copy.deepcopy(frame)

                # Byte Tracker推論
                _, bboxes, ids, scores = byte_tracker.inference(frame)

                elapsed_time = time.time() - start_time

                # Log detection info periodically
                if frame_id % 30 == 0:  # Log every 30 frames
                    logger.info(f"Frame {frame_id}: Detected {len(bboxes)} objects, inference time: {elapsed_time*1000:.2f}ms")

                # 検出情報描画
                debug_image = draw_tracking_info(
                    debug_image,
                    bboxes,
                    ids,
                    scores,
                    frame_id,
                    elapsed_time,
                )

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
