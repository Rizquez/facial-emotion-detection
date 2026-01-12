# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import FER_FILE, SEED, BATCH_SIZE, FER_IMAGE_SIZE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['load_fer_datasets']

def load_fer_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    # TODO: Documentation
    """
    df = pd.read_csv(FER_FILE)

    images = np.stack([_parse_pixels_to_image(pixel) for pixel in df['pixels'].values]).astype(np.float32)
    labels = df['emotion'].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.shuffle(buffer_size=2048, seed=SEED, reshuffle_each_iteration=True)

    n_row = len(df)
    n_train = int(0.8 * n_row)
    train_ds = ds.take(n_train)
    valid_ds = ds.skip(n_train)

    def preprocess(image, label):
        """
        # TODO: Documentation
        """
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, FER_IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label

    train_ds = train_ds.map(
        preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    )

    valid_ds = valid_ds.map(
        preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    )

    return train_ds, valid_ds

def _parse_pixels_to_image(pixels: str) -> 'ndarray':
    """
    # TODO: Documentation
    """
    array_pixels = np.fromstring(pixels, dtype=np.uint8, sep=' ')
    image = array_pixels.reshape(48, 48, 1)
    return image

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE