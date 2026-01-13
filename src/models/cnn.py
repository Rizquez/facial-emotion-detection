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
# TODO: Documentation
"""

KERNEL_SIZE = (3, 3)
"""
# TODO: Documentation
"""

SHAPE = (CK_IMAGE_SIZE[0], CK_IMAGE_SIZE[1], 1)
"""
# TODO: Documentation
"""

def build_ck_model() -> Sequential:
    """
    # TODO: Documentation
    """
    augmentation = Sequential([
        RandomRotation(factor=0.05),
        RandomFlip(mode='horizontal'),
        RandomBrightness(factor=0.1)
    ])

    model = Sequential([
        Input(shape=SHAPE),
        augmentation,

        Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=POOL_SIZE),
        Dropout(rate=0.25),

        Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=POOL_SIZE),
        Dropout(rate=0.25),

        Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=POOL_SIZE),
        Dropout(rate=0.25),

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
    # TODO: Documentation
    """
    model.fit(
        train_ds, 
        validation_data=valid_ds, 
        epochs=25
    )

    model.save_weights(CK_WEIGHTS_FILE)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE