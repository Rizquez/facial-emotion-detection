# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from typing import Tuple, TYPE_CHECKING
from keras.layers import Rescaling
from keras.utils import image_dataset_from_directory

if TYPE_CHECKING:
    import tensorflow as tf
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import IMAGE_SIZE, COLOR_MODE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['load_datasets']

DIRECTORY = 'CK+48'
"""
Root directory containing the image dataset organized by classes.

Each subdirectory within this directory represents a different class, and its name is used as a label when loading 
the dataset using `image_dataset_from_directory`.
"""

SEED = 42
"""
Seed used to ensure reproducibility in dataset partitioning.

This value is used during the division between training and validation to ensure that data separation is consistent 
between  runs.
"""

BATCH_SIZE = 32
"""
Number of samples included in each batch of the dataset.

This value controls how many images are processed simultaneously during training and validation.
"""

rescaling = Rescaling(1/255)
"""
Normalization layer used to rescale the pixel values of images.

The applied transformation converts the original values from the range [0, 255] to the range [0, 1], which facilitates 
model convergence.
"""

def load_datasets() -> Tuple['tf.data.Dataset', 'tf.data.Dataset']:
    """
    Load and prepare training and validation datasets from a directory structured by classes.

    This function uses `image_dataset_from_directory` to build two instances of `tf.data.Dataset` 
    from the same root directory, applying automatic partitioning into training and validation using 
    `validation_split`.

    **Process characteristics:**
        - Images are resized to the size defined in `IMAGE_SIZE`.
        - The color mode is controlled by `COLOR_MODE`.
        - Datasets are divided reproducibly using the `SEED` seed.
        - Each dataset is grouped into batches of size `BATCH_SIZE`.
        - Pixel values are normalized to the range [0, 1] using a `Rescaling` layer.

    **Expected directory structure:**
        - The root directory must contain one subfolder per class.
        - The name of each subfolder is used as a label.

    Returns:
        Tuple:
            Tuple with the training dataset and the validation dataset in that order.
    """
    training = image_dataset_from_directory(
        directory=DIRECTORY,
        validation_split=0.2,
        subset='training',
        seed=SEED,
        image_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE
    )

    validation = image_dataset_from_directory(
        directory=DIRECTORY,
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        image_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE
    )

    training = _normalize_dataset(training)

    validation = _normalize_dataset(validation)

    return training, validation

def _normalize_dataset(dataset: 'tf.data.Dataset') -> 'tf.data.Dataset':
    """
    Applies pixel value normalization to a TensorFlow dataset.

    This function transforms each batch of the dataset by applying a layer of type `Rescaling` 
    only to the images, preserving the labels without modification.

    The input dataset is expected to produce elements with the structure `(x, y)`, where:
        - `x` represents a batch of images.
        - `y` represents the corresponding labels.

    The resulting transformation maintains the same dataset structure, but with normalized image values.

    Args:
        dataset (tf.data.Dataset):
            Dataset that produces batches of the form `(x, y)`.

    Returns:
        tf.data.Dataset:
            New dataset that produces batches, with the normalized images and labels intact.
    """
    return dataset.map(lambda x_batch, y_batch: (rescaling(x_batch), y_batch))

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE