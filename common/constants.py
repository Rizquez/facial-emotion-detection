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

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'keras.weights.h5')
"""
Absolute path where the weights of the trained model are stored and from where they are loaded.
"""

EMOTION_LABELS = [
    'anger',
    'contempt',
    'disgust',
    'fear',
    'happy',
    'sadness',
    'surprise'
]
"""
List of emotion labels associated with the classes in the model.
"""

IMAGE_SIZE = (48, 48)
"""
Target size to which all input images are resized, The tuple represents (height, width) in pixels. 
"""

COLOR_MODE = 'grayscale'
"""
Color mode used when loading images.
"""

INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
"""
Expected form of the neural model input.
"""

NUM_CLASSES = len(EMOTION_LABELS)
"""
Total number of classes to be predicted by the model.
"""

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE