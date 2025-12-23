# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from keras.layers import Rescaling
from keras.utils import image_dataset_from_directory
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import IMAGE_SIZE, COLOR_MODE, SHAPE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['load_datasets']

CK_PLUS_PATH = 'assets/CK+48'
"""
Root directory containing the image dataset organized by classes.
"""

FER_FILE = 'assets/FER.csv'
"""
Path to the CSV file containing the FER (Facial Expression Recognition) dataset.
"""

SEED = 42
"""
Seed used to ensure reproducibility in dataset partitioning.
"""

BATCH_SIZE = 32
"""
Number of samples included in each batch of the dataset.
"""

rescaling = Rescaling(1/255)
"""
Normalization layer used to rescale the pixel values of images.

The applied transformation converts the original values from the 
range [0, 255] to the range [0, 1], which facilitates model convergence.
"""

def load_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load, prepare, and combine the CK+ and FER datasets for training and validation.

    This function builds two independent datasets from different sources:
        - The CK+ dataset, loaded from a directory structured by classes.
        - The FER dataset, loaded from a CSV file and converted to `tf.data.Dataset`.

    Both sets are processed uniformly and then concatenated to form a single training dataset 
    and a single validation dataset.

    **Detailed process:**
        - The CK+ data is loaded from disk, applying automatic partitioning into training and validation.
        - The pixel values of the CK+ datasets are normalized to the range [0, 1].
        - The FER datasets are loaded, which already returns the normalized training and validation sets.
        - The corresponding CK+ and FER datasets are concatenated:
            * CK+ training + FER training.
            * CK+ validation + FER validation.

    **Characteristics of the resulting dataset:**
        - Images are resized to the size defined in `IMAGE_SIZE`.
        - The color mode is controlled by `COLOR_MODE`.
        - The data is grouped into batches of size `BATCH_SIZE`.
        - Pixel values are normalized to the range [0, 1].
        - The structure of each element is `(x, y)`, where `x` are images and `y` are class labels.

    Returns:
        Tuple:
            Tuples containing the training and validation CK+ and FER combined datasets.
    """
    train_ck = image_dataset_from_directory(
        directory=CK_PLUS_PATH,
        validation_split=0.2,
        subset='training',
        seed=SEED,
        image_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE
    )

    valid_ck = image_dataset_from_directory(
        directory=CK_PLUS_PATH,
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        image_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE
    )

    train_ck = _normalize_dataset(train_ck)
    valid_ck = _normalize_dataset(valid_ck)

    train_fer, valid_fer = _load_fer_dataset()

    train_ds = train_ck.concatenate(train_fer)
    valid_ds = valid_ck.concatenate(valid_fer)

    return train_ds, valid_ds

def _load_fer_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load and prepare the FER dataset from a CSV file.

    This function reads the file defined in `FER_FILE`, reconstructs the images from
    the pixel values stored as text, and generates a `tf.data.Dataset` with their
    corresponding labels.

    **Preparation process:**
        - The CSV file is read using pandas.
        - Pixel values are converted to NumPy arrays and resized using `SHAPE`.
        - Image and label arrays are constructed.
        - A `tf.data.Dataset` is created from this data.
                - The dataset is shuffled in a reproducible manner using the `SEED` seed.
                - The set is divided into training (80%) and validation (20%).
                - The resulting datasets are grouped into batches of size `BATCH_SIZE`.
                - Pixel values are normalized to the range [0, 1].

    Returns:
        Tuple:
            Tuples containing the training and validation FER datasets.
    """
    df = pd.read_csv(FER_FILE)

    images = []
    labels = []

    for _, row in df.iterrows():
        pixels = np.array(row['pixels'].split(), dtype=np.uint8)
        image = pixels.reshape(SHAPE)
        images.append(image)
        labels.append(row['emotion'])

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1024, seed=SEED)

    train_size = int(0.8 * len(images))
    train_fer = dataset.take(train_size)
    valid_fer = dataset.skip(train_size)

    train_fer = train_fer.batch(BATCH_SIZE)
    valid_fer = valid_fer.batch(BATCH_SIZE)

    train_fer = _normalize_dataset(train_fer)
    valid_fer = _normalize_dataset(valid_fer)

    return train_fer, valid_fer

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
    return dataset.map(lambda x_batch, y_batch: (rescaling(x_batch), y_batch))

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE