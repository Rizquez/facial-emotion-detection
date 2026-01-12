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
from common.constants import CK_PATH, CK_COLOR_MODE, CK_IMAGE_SIZE, SEED, BATCH_SIZE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['load_ck_datasets']

_rescaling = Rescaling(1 / 255)
"""
# TODO: Documentation
"""

def load_ck_datasets() -> Tuple['tf.data.Dataset', 'tf.data.Dataset']:
    """
    # TODO: Documentation
    """
    train_ds = image_dataset_from_directory(
        directory=CK_PATH,
        validation_split=0.2,
        subset='training',
        seed=SEED,
        image_size=CK_IMAGE_SIZE,
        color_mode=CK_COLOR_MODE,
        batch_size=BATCH_SIZE,
    )

    valid_ds = image_dataset_from_directory(
        directory=CK_PATH,
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        image_size=CK_IMAGE_SIZE,
        color_mode=CK_COLOR_MODE,
        batch_size=BATCH_SIZE,
    )

    train_ds = _normalize_dataset(train_ds)

    valid_ds = _normalize_dataset(valid_ds)

    return train_ds, valid_ds

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
    return dataset.map(lambda x_batch, y_batch: (_rescaling(x_batch), y_batch))

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE