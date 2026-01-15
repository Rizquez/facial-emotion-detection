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
Absolute path to the project's root directory.
"""

CK_PATH = os.path.join(PROJECT_ROOT, 'assets', 'CK+48')
"""
Path to the CK+ dataset directory (preprocessed/organized by folders).

Each folder represents an emotion class and contains images from that category.
"""

FER_FILE = os.path.join(PROJECT_ROOT, 'assets', 'FER.csv')
"""
Path to the CSV file of the FER2013 dataset.
"""

MP_FACE_DETECTOR_MODEL = os.path.join(PROJECT_ROOT, 'assets', 'blaze_face_short_range.tflite')
"""
Path to the TFLite model used by MediaPipe to detect faces.
"""

CK_WEIGHTS_FILE = os.path.join(PROJECT_ROOT, 'ck.weights.h5')
"""
Path to the weight file of the model trained with CK+.
"""

FER_WEIGHTS_FILE = os.path.join(PROJECT_ROOT, 'fer.weights.h5')
"""
Path to the weight file of the model trained with FER2013.
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
List of tags (classes) for the CK+ dataset.
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
List of tags (classes) for the FER2013 dataset.
"""

CK_IMAGE_SIZE = (48, 48)
"""
Target image size for the CK+ pipeline.


The images are resized to 48x48 to:

- Reduce computational cost
- Maintain consistency with the preprocessed dataset
- Feed the CNN model defined for this resolution
"""

CK_COLOR_MODE = 'grayscale'
"""
Color mode used in the CK+ pipeline.

CK+ is processed in grayscale:

- Reduces dimensionality (1 channel)
- Accelerates training/inference
- Is sufficient for facial expressions in many scenarios
"""

FER_IMAGE_SIZE = (96, 96)
"""
Target size for the FER2013 pipeline.

Although FER2013 starts at 48x48 in grayscale, in this project:

- It is converted to RGB (3 channels)
- It is resized to 96x96

To be compatible and take advantage of a pre-trained backbone (MobileNetV2).
"""

SEED = 42
"""
Global seed for reproducibility.

Used for:

- Training/validation splits
- Dataset shuffling
- Any controllable random operation

Note: in Deep Learning, perfect reproducibility may vary by backend/hardware, but this seed helps stabilize results.
"""

BATCH_SIZE = 32
"""
Batch size for training and dataset creation.

A value of 32 is usually a reasonable balance between:

- Gradient stability
- Memory consumption
- Training speed

It can be adjusted according to available hardware (CPU/GPU and RAM/VRAM).
"""

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE