# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from src.models import build_ck_model, build_fer_model
from common.constants import (
    MP_FACE_DETECTOR_MODEL,
    CK_COLOR_MODE, CK_IMAGE_SIZE, CK_EMOTION_LABELS, CK_WEIGHTS_FILE,
    FER_IMAGE_SIZE, FER_EMOTION_LABELS, FER_WEIGHTS_FILE
)
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['activate_webcam']

COLOR = (0, 255, 0)
"""
# TODO: Documentation
"""

MARGIN = 0.20
"""
# TODO: Documentation
"""

def activate_webcam(source: Literal['ck', 'fer'] = 'ck') -> None:
    """
    # TODO: Documentation
    """
    source = source.lower().strip()
    if source not in ('ck', 'fer'):
        raise ValueError('The `source` parameter must be equal to `ck` or `fer`')
    
    if source == 'ck':
        model = build_ck_model()
        model.load_weights(CK_WEIGHTS_FILE)
        emotion_labels = CK_EMOTION_LABELS
        expected_color_mode = CK_COLOR_MODE
    else:
        model = build_fer_model()
        model.load_weights(FER_WEIGHTS_FILE)
        emotion_labels = FER_EMOTION_LABELS
        expected_color_mode = 'bgr'

    probs_hist = deque(maxlen=8)

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MP_FACE_DETECTOR_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.6,
    )

    detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        raise RuntimeError("The webcam could not be accessed")
    
    while True:
        frame_read_success, frame_bgr = video_capture.read()

        if not frame_read_success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.perf_counter() * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if not result.detections:
            probs_hist.clear()
            cv2.imshow(...)
            continue
        
        grayscale_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        h_image, w_image = frame_bgr.shape[:2]

        for detection in result.detections:
            bbox = detection.bounding_box
            origin_x, origin_y, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            x1 = max(0, int(origin_x - width * MARGIN))
            y1 = max(0, int(origin_y - height * MARGIN))
            x2 = min(w_image, int(origin_x + width * (1 + MARGIN)))
            y2 = min(h_image, int(origin_y + height * (1 + MARGIN)))

            face_gray = grayscale_frame[y1:y2, x1:x2]
            face_bgr  = frame_bgr[y1:y2, x1:x2]

            if face_gray.size == 0 or face_bgr.size == 0:
                continue

            if expected_color_mode == CK_COLOR_MODE:
                model_input = _preprocess_face_ck(face_gray)
            else:
                model_input = _preprocess_face_fer(face_bgr)

            predictions = model.predict(model_input, verbose=0)[0]
            probs_hist.append(predictions)
            avg = np.mean(probs_hist, axis=0)

            predicted_emotion = emotion_labels[int(np.argmax(avg))]

            cv2.rectangle(
                img=frame_bgr, 
                pt1=(x1, y1), 
                pt2=(x2, y2), 
                color=COLOR, 
                thickness=2
            )

            cv2.putText(
                img=frame_bgr, 
                text=predicted_emotion, 
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.9, 
                color=COLOR, 
                thickness=2
            )


        cv2.imshow(f"Emotion detection ({source})", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()  
    video_capture.release()
    cv2.destroyAllWindows()

def _preprocess_face_ck(gray_face: 'ndarray') -> 'ndarray':
    """
    # TODO: Documentation
    """
    resized = cv2.resize(src=gray_face, dsize=CK_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, CK_IMAGE_SIZE[0], CK_IMAGE_SIZE[1], 1)

def _preprocess_face_fer(bgr_face: 'ndarray') -> 'ndarray':
    """
    # TODO: Documentation
    """
    resized = cv2.resize(src=bgr_face, dsize=FER_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(src=resized, code=cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.astype(np.float32), axis=0)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE