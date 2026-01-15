# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
from keras import optimizers
from keras.models import Sequential
from keras.layers import (
    Input, 
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
    RandomRotation, RandomFlip, RandomBrightness, 
    BatchNormalization, Dropout, Dense
)

if TYPE_CHECKING:
    import tensorflow as tf
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import CK_IMAGE_SIZE, CK_EMOTION_LABELS, CK_WEIGHTS_FILE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['build_ck_model', 'train_ck_model']

POOL_SIZE = (2, 2)
"""
Size of the pooling window used in MaxPooling2D layers.

A 2x2 pooling halves the spatial resolution of the activation maps, reducing computational cost and helping to make 
the model more robust to small spatial variations in the images.
"""

KERNEL_SIZE = (3, 3)
"""
Kernel size used in convolutional layers.

A 3x3 kernel is a standard choice in CNNs, as it allows local features (edges, textures) to be captured with a good 
balance between expressive power and computational cost.
"""

SHAPE = (CK_IMAGE_SIZE[0], CK_IMAGE_SIZE[1], 1)
"""
Model input format for the CK+ dataset.
"""

def build_ck_model() -> Sequential:
    """
    Build and compile the CNN model for emotion classification with CK+.

    The model follows a classic convolutional architecture:
        - Conv2D blocks + BatchNormalization + MaxPooling + Dropout
        - Data augmentation applied directly to the input
        - Global pooling to reduce parameters
        - Final dense layers for multi-class classification

    The model is optimized for small images (48x48) in grayscale and for 
    efficient training in environments without a GPU.

    Returns:
        Sequential:
            Compiled model ready for training or inference.
    """
    # Data augmentation block to improve generalization
    augmentation = Sequential([
        RandomRotation(factor=0.05),    # Small random rotations
        RandomFlip(mode='horizontal'),  # Horizontal flip
        RandomBrightness(factor=0.1)    # Slight variations in brightness
    ])

    model = Sequential([
        # Input layer with data augmentation
        Input(shape=SHAPE),
        augmentation,

        # First convolutional block
        Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=POOL_SIZE),
        Dropout(rate=0.25),

        # Second convolutional block
        Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=POOL_SIZE),
        Dropout(rate=0.25),

        # Third convolutional block
        Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=POOL_SIZE),
        Dropout(rate=0.25),

        # Overall reduction and classification
        GlobalAveragePooling2D(),
        Dense(units=128, activation='relu'),
        Dropout(rate=0.5),
        Dense(units=len(CK_EMOTION_LABELS), activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_ck_model(model: Sequential, train_ds: 'tf.data.Dataset', valid_ds: 'tf.data.Dataset') -> None:
    """
    Train the CNN model for the CK+ dataset and save the trained weights.

    **This function:**
        - Trains the model for a fixed number of epochs.
        - Evaluates performance on the validation set at each epoch.
        - Saves the final model weights for later reuse.

    Args:
        model (Sequential):
            Previously built and compiled CNN model.
        train_ds (tf.data.Dataset):
            Training dataset.
        valid_ds (tf.data.Dataset):
            Validation dataset.
    """
    model.fit(
        train_ds, 
        validation_data=valid_ds, 
        epochs=25
    )

    model.save_weights(CK_WEIGHTS_FILE)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE