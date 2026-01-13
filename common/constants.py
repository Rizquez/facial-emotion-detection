# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import os
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
# Get listed here!
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""
# TODO: Documentation
"""

CK_PATH = os.path.join(PROJECT_ROOT, 'assets', 'CK+48')
"""
# TODO: Documentation
"""

FER_FILE = os.path.join(PROJECT_ROOT, 'assets', 'FER.csv')
"""
# TODO: Documentation
"""

MP_FACE_DETECTOR_MODEL = os.path.join(PROJECT_ROOT, 'assets', 'blaze_face_short_range.tflite')
"""
# TODO: Documentation
"""

CK_WEIGHTS_FILE = os.path.join(PROJECT_ROOT, 'ck.weights.h5')
"""
# TODO: Documentation
"""

FER_WEIGHTS_FILE = os.path.join(PROJECT_ROOT, 'fer.weights.h5')
"""
# TODO: Documentation
"""

CK_EMOTION_LABELS = [
    'anger',
    'contempt',
    'disgust',
    'fear',
    'happy',
    'sadness',
    'surprise'
]
"""
# TODO: Documentation
"""

FER_EMOTION_LABELS = [
    'anger',
    'disgust',
    'fear',
    'happy',
    'sadness',
    'surprise',
    'neutral'
]
"""
# TODO: Documentation
"""

CK_IMAGE_SIZE = (48, 48)
"""
# TODO: Documentation
"""

CK_COLOR_MODE = 'grayscale'
"""
# TODO: Documentation
"""

FER_IMAGE_SIZE = (96, 96)
"""
# TODO: Documentation
"""

SEED = 42
"""
# TODO: Documentation
"""

BATCH_SIZE = 32
"""
# TODO: Documentation
"""

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE