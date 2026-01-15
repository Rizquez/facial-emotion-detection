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
Color (BGR) used to draw annotations on the frame.

OpenCV works in BGR format by default, so (0, 255, 0) represents the color green.
"""

MARGIN = 0.20
"""
Additional margin applied to the face bounding box.

MediaPipe returns a box fitted to the face; this margin expands the cropped region to include some context. 

This often improves the stability of the emotion classifier, especially when the detector crops too close.
"""

def activate_webcam(source: Literal['ck', 'fer']) -> None:
    """
    Activate the webcam and run facial emotion detection in real time.

    **The general flow is:**
        1) Select the model according to the `source` (`ck` or `fer`) and load its weights.
        2) Initialize a face detector (MediaPipe FaceDetector).
        3) Capture frames from the webcam with OpenCV.
        4) Detect faces in each frame.
        5) Crop the detected face and preprocess it according to the chosen model.
        6) Infer the emotion and display it on the image.
        7) Smooth predictions with a time window (moving average) to reduce flickering.

    Args:
        source (Literal['ck', 'fer'], optional):
            Indicates the pipeline to use: ck for CNN trained with CK+ and fer for MobileNetV2 trained with FER2013.

    Raises:
        ValueError:
            If `source` is neither ck nor fer.
        RuntimeError:
            If the webcam cannot be accessed.
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
        expected_color_mode = 'bgr' # The FER pipeline works internally with BGR frames and converts to RGB later

    # Short history of probabilities for temporal smoothing (reduces flickering)
    probs_hist = deque(maxlen=8)

    # Configuring MediaPipe's face detector in video mode
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MP_FACE_DETECTOR_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.6,
    )

    detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    # Camera access (index=0 is usually the main webcam)
    video_capture = cv2.VideoCapture(index=0)
    if not video_capture.isOpened():
        raise RuntimeError("The webcam could not be accessed")
    
    # MediaPipe in video mode requires increasing timestamps (in milliseconds)
    start = time.monotonic()
    last_ts = -1
    
    while True:
        ok, frame_bgr = video_capture.read()
        if not ok:
            break
        
        # Timestamp in milliseconds since the start of capture
        timestamp_ms = int((time.monotonic() - start) * 1000)

        # Ensures that the timestamp always increases (requirement for the detector in video mode)
        if timestamp_ms <= last_ts:
            timestamp_ms = last_ts + 1
        last_ts = timestamp_ms

        # MediaPipe expects RGB, so we convert from BGR
        frame_rgb = cv2.cvtColor(src=frame_bgr, code=cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Face detection for the current frame
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # If there are no faces, we clear the probability history (to avoid dragging)
        if not result.detections:
            probs_hist.clear()
            continue
        
        # We precompute the gray version for the CK pipeline
        grayscale_frame = cv2.cvtColor(src=frame_bgr, code=cv2.COLOR_BGR2GRAY)

        h_image, w_image = frame_bgr.shape[:2]

        for detection in result.detections:
            bbox = detection.bounding_box
            origin_x, origin_y, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            
            # Convert bbox to a square crop with margin, centered on the face
            center_x = origin_x + width / 2.0
            center_y = origin_y + height / 2.0
            side = max(width, height) * (1.0 + MARGIN)

            x1 = max(0, int(center_x - side / 2))
            y1 = max(0, int(center_y - side / 2))
            x2 = min(w_image, int(center_x + side / 2))
            y2 = min(h_image, int(center_y + side / 2))

            # Face cutout in gray (CK) and in BGR (FER)
            face_gray = grayscale_frame[y1:y2, x1:x2]
            face_bgr  = frame_bgr[y1:y2, x1:x2]

            # Defensive validation: avoid errors if the cutout is empty
            if face_gray.size == 0 or face_bgr.size == 0:
                continue
            
            # Preprocessing according to the selected pipeline
            if expected_color_mode == CK_COLOR_MODE:
                model_input = _preprocess_face_ck(face_gray)
            else:
                model_input = _preprocess_face_fer(face_bgr)

            # Inference: we obtain probability distribution by emotion
            predictions = model.predict(model_input, verbose=0)[0]

            # Temporal smoothing: moving average of the last N predictions
            probs_hist.append(predictions)
            avg = np.mean(probs_hist, axis=0)

            predicted_emotion = emotion_labels[int(np.argmax(avg))]

            # Drawn from the bounding box
            cv2.rectangle(
                img=frame_bgr, 
                pt1=(x1, y1), 
                pt2=(x2, y2), 
                color=COLOR, 
                thickness=2
            )

            # Emotional text slightly above the rectangle
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

        # Exit the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release of resources
    detector.close()  
    video_capture.release()
    cv2.destroyAllWindows()

def _preprocess_face_ck(gray_face: 'ndarray') -> 'ndarray':
    """
    Preprocesses a face crop for the model trained with CK+.

    The CK model expects:
        - Grayscale image.
        - Size 48x48.
        - Normalization [0, 1].
        - Shape (1, H, W, 1) for batch size 1.

    Args:
        gray_face (ndarray):
            Grayscale face crop.

    Returns:
        ndarray:
            Tensor with shape (1, 48, 48, 1) and normalized values.
    """
    resized = cv2.resize(
        src=gray_face, 
        dsize=CK_IMAGE_SIZE, 
        interpolation=cv2.INTER_AREA # Suitable for downsizing (downsampling)
    )
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, CK_IMAGE_SIZE[0], CK_IMAGE_SIZE[1], 1)

def _preprocess_face_fer(bgr_face: 'ndarray') -> 'ndarray':
    """
    Preprocesses a facial crop for the model trained with FER2013 (MobileNetV2).

    Here, the BGR crop is first converted to grayscale and then to RGB. This maintains 
    consistency with the origin of the FER2013 dataset (grayscale), but allows feeding 
    a pre-trained backbone that requires 3 channels.

    **The FER model expects:**
        - RGB image (3 channels).
        - Size defined by FER_IMAGE_SIZE (default 96x96).
        - Shape (1, H, W, 3) for batch size 1.

    Args:
        bgr_face (ndarray):
            Face crop in BGR format (OpenCV).

    Returns:
        ndarray:
            Tensor with shape (1, H, W, 3) and type float32.
    """
    gray = cv2.cvtColor(src=bgr_face, code=cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        src=gray, 
        dsize=FER_IMAGE_SIZE, 
        interpolation=cv2.INTER_LINEAR # Suitable for general rescaling
    )
    rgb = cv2.cvtColor(src=resized, code=cv2.COLOR_GRAY2RGB)
    return np.expand_dims(rgb.astype(np.float32), axis=0)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE