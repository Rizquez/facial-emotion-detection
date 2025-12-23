# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, RandomRotation, RandomFlip, RandomBrightness

if TYPE_CHECKING:
    import tensorflow as tf
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import SHAPE, NUM_CLASSES, MODEL_PATH
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['build_model', 'train_model']

KERNEL_SIZE = (3, 3)
"""
Kernel size used in convolutional layers.
"""

POOL_SIZE = (2, 2)
"""
Size of the pooling window used in max pooling layers.
"""

def build_model() -> Sequential:
    """
    Build and compile a Convolutional Neural Network (CNN) model for facial emotion detection.

    The model incorporates a data augmentation phase that is applied exclusively during training, 
    followed by a lightweight CNN architecture designed to balance accuracy and computational efficiency.

    Data augmentation techniques used:
        - Random rotation to simulate variations in head position.
        - Horizontal flipping to improve left-right invariance.
        - Brightness adjustment to simulate different lighting conditions.

    The model is compiled using the Adam optimizer with a reduced learning rate, allowing for more stable 
    convergence during training.

    Returns:
        Sequential:
            Keras model compiled and ready to be trained.
    """
    augmentation = Sequential([
        RandomRotation(factor=0.5),
        RandomFlip(mode='horizontal'),
        RandomBrightness(factor=0.1),
    ])

    model = Sequential([
        Input(shape=SHAPE),

        augmentation, # Data increase applied only during training

        Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu'),
        MaxPooling2D(pool_size=POOL_SIZE),

        Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu'),
        MaxPooling2D(pool_size=POOL_SIZE),

        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model: Sequential, train_ds: 'tf.data.Dataset', valid_ds: 'tf.data.Dataset') -> None:
    """
    Train a Keras model using training and validation datasets.

    This function executes the training process by calling `model.fit`, evaluating the model 
    on the validation set at each epoch and save the model on a specific path.

    Args:
        model (Sequential):
            Previously built and compiled Keras model.
        train_ds (tf.data.Dataset):
            Training dataset that produces batches of images and labels.
        valid_ds (tf.data.Dataset):
            Validation dataset used to evaluate model performance during training.
    """
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10
    )

    model.save_weights(filepath=MODEL_PATH)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE