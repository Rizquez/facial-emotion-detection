# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
from typing import Literal, Any, Union, TYPE_CHECKING
from keras.applications.vgg19 import preprocess_input


if TYPE_CHECKING:
    from numpy import ndarray
    from keras.src.backend.openvino.core import OpenVINOKerasTensor
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from src.models import build_ck_model, build_fer_model
from common.constants import (
    CASCADE_FILE,
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

def activate_webcam(mode: Literal['ck', 'fer'] = 'ck') -> None:
    """
    # TODO: Documentation
    """
    mode = mode.lower().strip()
    if mode not in ('ck', 'fer'):
        raise ValueError('The mode parameter must be equal to `ck` or `fer`')
    
    if mode == 'ck':
        model = build_ck_model()
        model.load_weights(CK_WEIGHTS_FILE)
        emotion_labels = CK_EMOTION_LABELS
        expected_color_mode = CK_COLOR_MODE
    else:
        model = build_fer_model()
        model.load_weights(FER_WEIGHTS_FILE)
        emotion_labels = FER_EMOTION_LABELS
        expected_color_mode = 'bgr'

    face_detector = cv2.CascadeClassifier(CASCADE_FILE)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("The webcam could not be accessed")
        return
    
    while True:
        frame_read_success, frame_bgr = video_capture.read()
        if not frame_read_success:
            break

        grayscale_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(
            image=grayscale_frame,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x_coord, y_coord, width, height) in detected_faces:

            face_gray = grayscale_frame[
                y_coord : y_coord + height,
                x_coord : x_coord + width,
            ]

            face_bgr = frame_bgr[
                y_coord : y_coord + height,
                x_coord : x_coord + width,
            ]

            if expected_color_mode == CK_COLOR_MODE:
                model_input = _preprocess_face_ck(face_gray)
            else:
                model_input = _preprocess_face_fer(face_bgr)

            predictions = model.predict(model_input, verbose=0)
            predicted_emotion = emotion_labels[int(np.argmax(predictions))]

            cv2.rectangle(
                img=frame_bgr,
                pt1=(x_coord, y_coord),
                pt2=(x_coord + width, y_coord + height),
                color=COLOR,
                thickness=2,
            )

            cv2.putText(
                img=frame_bgr,
                text=predicted_emotion,
                org=(x_coord, y_coord - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=COLOR,
                thickness=2,
            )

        cv2.imshow(f"Emotion detection ({mode})", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()

def _preprocess_face_ck(gray_face: 'ndarray') -> 'ndarray':
    """
    # TODO: Documentation
    """
    resized_face = cv2.resize(src=gray_face, dsize=CK_IMAGE_SIZE)
    normalized_face = resized_face.astype(np.float32) / 255

    model_input = np.reshape(normalized_face, shape=(1, CK_IMAGE_SIZE[0], CK_IMAGE_SIZE[1], 1))
    return model_input


def _preprocess_face_fer(bgr_face: 'ndarray') -> Union[Any, OpenVINOKerasTensor]:
    """
    # TODO: Documentation
    """
    resized_face = cv2.resize(src=bgr_face, dsize=FER_IMAGE_SIZE)
    rgb_face = cv2.cvtColor(src=resized_face, code=cv2.COLOR_BGR2RGB)

    float_face = rgb_face.astype(np.float32)
    batch_face = np.expand_dims(float_face, axis=0)

    model_input = preprocess_input(batch_face)
    return model_input

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE