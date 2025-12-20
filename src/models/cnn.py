# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

if TYPE_CHECKING:
    import tensorflow as tf
    from keras.callbacks import History
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import IMAGE_SIZE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['build_model', 'train_model']

INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
"""
Expected form of the neural model input.

It is defined based on the image size and the number of channels.

The last value (`1`) corresponds to grayscale images.
"""

CONV_FILTERS = (32, 64)
"""
Number of filters used in the convolutional layers of the model.

Each value in the tuple corresponds to the number of filters in a convolutional layer, in the same order in which 
they are defined within the model.
"""

DENSE_UNITS = 128
"""
Number of neurons in the dense intermediate layer of the model.

This layer is used to combine and abstract the features extracted by the convolutional layers before the output layer.
"""

KERNEL_SIZE = (3, 3)
"""
Kernel size used in convolutional layers.

The tuple represents (height, width) of the convolutional filter applied to the input images.
"""

POOL_SIZE = (2, 2)
"""
Size of the pooling window used in max pooling layers.

The tuple represents (height, width) of the region over which the spatial reduction operation is applied.
"""

ACTIVATION_CONV = 'relu'
"""
Activation function used in convolutional layers.

The ReLU (Rectified Linear Unit) function introduces non-linearity into the model and helps mitigate the vanishing 
gradient problem.
"""

ACTIVATION_DENSE = 'relu'
"""
Activation function used in the dense intermediate layer.

It is used to introduce non-linearity after the feature extraction phase and before the final classification layer.
"""

NUM_CLASSES = 7
"""
Total number of classes to be predicted by the model.

This value must match the number of categories present in the dataset. It is used to define the number of neurons 
in the output layer of the model.
"""

OPTIMIZER = 'adam'
"""
Optimizer used during the model training process.

The Adam optimizer combines momentum and learning rate adaptation techniques to achieve efficient convergence across 
a wide variety of problems.
"""

LOSS = 'sparse_categorical_crossentropy'
"""
Loss function used for model training.

This function is suitable when labels are encoded as integers rather than one-hot vectors.
"""

METRICS = ['accuracy']
"""
List of metrics evaluated during training and validation.

- The `accuracy` metric measures the proportion of correct predictions made by the model.
"""

EPOCHS = 5
"""
Number of epochs used during model training.

One epoch corresponds to one complete pass through the training dataset.
"""

def build_model() -> Sequential:
    """
    Build and compile a convolutional model for image classification.

    The model is designed to work with grayscale images and produce a 
    probability distribution over `NUM_CLASSES` classes.

    Returns:
        Sequential:
            Keras model compiled and ready to be trained.
    """
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        Conv2D(CONV_FILTERS[0], KERNEL_SIZE, activation=ACTIVATION_CONV),
        MaxPooling2D(POOL_SIZE),

        Conv2D(CONV_FILTERS[1], KERNEL_SIZE, activation=ACTIVATION_CONV),
        MaxPooling2D(POOL_SIZE),

        Flatten(),
        Dense(DENSE_UNITS, activation=ACTIVATION_DENSE),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=METRICS
    )

    return model

def train_model(model: Sequential, training: 'tf.data.Dataset', validation: 'tf.data.Dataset') -> 'History':
    """
    Train a Keras model using training and validation datasets.

    This function executes the training process by calling `model.fit`, using the number of epochs defined 
    in `EPOCHS` and evaluating the model on the validation set at each epoch.

    Args:
        model (Sequential):
            Previously built and compiled Keras model.
        training (tf.data.Dataset):
            Training dataset that produces batches of images and labels.
        validation (tf.data.Dataset):
            Validation dataset used to evaluate model performance
            during training.

    Returns:
        History:
            Object containing the history of metrics and loss values recorded during training.
    """
    return model.fit(
        training,
        validation_data=validation,
        epochs=EPOCHS
    )

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE