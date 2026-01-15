# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from typing import Tuple
from keras.layers import Rescaling
from keras.utils import image_dataset_from_directory
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import CK_PATH, CK_COLOR_MODE, CK_IMAGE_SIZE, CK_EMOTION_LABELS, SEED, BATCH_SIZE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['load_ck_datasets']

_rescaling = Rescaling(1 / 255)
"""
Image normalization layer.

This layer scales the input pixel values from the range [0, 255] to the range [0.0, 1.0]. 

It is applied as part of the dataset preprocessing to improve numerical stability and the model training process.
"""

def load_ck_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load and prepare the CK+ dataset training and validation datasets.

    **This function:**
        - Loads images from the CK+ directory organized by folders (one per emotion).
        - Divides the dataset into training and validation using a fixed ratio (80/20).
        - Resizes the images to the size defined for CK+.
        - Converts the images to grayscale.
        - Normalizes the pixel values.
        - Applies shuffling, caching, and prefetching to improve performance.

    Returns:
        Tuple:
            - train_ds: Training dataset.
            - valid_ds: Validation dataset.

    """
    # Training dataset (80% of the data)
    train_ds = image_dataset_from_directory(
        directory=CK_PATH,
        validation_split=0.2,
        subset='training',
        seed=SEED,
        image_size=CK_IMAGE_SIZE,
        color_mode=CK_COLOR_MODE,
        batch_size=BATCH_SIZE,
        class_names=CK_EMOTION_LABELS
    )

    # Validation dataset (20% of the data)
    valid_ds = image_dataset_from_directory(
        directory=CK_PATH,
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        image_size=CK_IMAGE_SIZE,
        color_mode=CK_COLOR_MODE,
        batch_size=BATCH_SIZE,
        class_names=CK_EMOTION_LABELS
    )

    # Normalization of pixel values (0–255 -> 0.0–1.0)
    train_ds = _normalize_dataset(train_ds)
    valid_ds = _normalize_dataset(valid_ds)

    # Shuffling the training dataset to avoid bias due to loading order
    train_ds = train_ds.shuffle(buffer_size=1000, seed=SEED, reshuffle_each_iteration=True)

    # Cache and prefetch to improve performance during training
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds

def _normalize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
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