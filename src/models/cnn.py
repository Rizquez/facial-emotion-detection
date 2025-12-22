# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, RandomRotation, RandomFlip, RandomBrightness

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

ROTATION = 0.5
"""
Defines the maximum rotation factor applied during data augmentation.

A value of 0.5 allows for random rotations of up to approximately 50% of a full turn, which helps the model to be 
more robust against variations in head orientation and small misalignments in faces.
"""

BRIGHTNESS = 0.1
"""
Defines the range of brightness variation applied during data augmentation.

A value of 0.1 introduces small random modifications to the brightness of the images, simulating different lighting 
conditions and improving the model's generalization ability in real environments.
"""

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

EPOCHS = 10
"""
Number of epochs used during model training.

One epoch corresponds to one complete pass through the training dataset.
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
        RandomRotation(ROTATION),
        RandomFlip('horizontal'),
        RandomBrightness(BRIGHTNESS),
    ])

    model = Sequential([
        Input(shape=INPUT_SHAPE),

        augmentation, # Data increase applied only during training

        Conv2D(CONV_FILTERS[0], KERNEL_SIZE, activation=ACTIVATION_CONV),
        MaxPooling2D(POOL_SIZE),

        Conv2D(CONV_FILTERS[1], KERNEL_SIZE, activation=ACTIVATION_CONV),
        MaxPooling2D(POOL_SIZE),

        Flatten(),
        Dense(DENSE_UNITS, activation=ACTIVATION_DENSE),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
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