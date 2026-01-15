# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import pandas as pd
import tensorflow as tf
from typing import Tuple
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import FER_FILE, SEED, BATCH_SIZE, FER_IMAGE_SIZE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['load_fer_datasets']

def load_fer_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load and prepare the training, validation, and test datasets from the FER2013 dataset.

    **This function:**
        - Reads the CSV file from the FER2013 dataset.
        - Divides the data according to the `Usage` column into:
            * Training → training
            * PublicTest → validation
            * PrivateTest → test
        - Converts images stored as pixel strings into tensors.
        - Resizes and adapts images for use with a MobileNetV2-based model.
        - Builds efficient pipelines with `tf.data` (shuffle, batch, cache, and prefetch).

    Returns:
        Tuple:
            - train_ds: Training dataset.
            - valid_ds: Validation dataset.
            - test_ds: Test dataset.
    """
    df = pd.read_csv(FER_FILE)

    train_df = df[df['Usage'] == 'Training']
    valid_df = df[df['Usage'] == 'PublicTest']
    test_df  = df[df['Usage'] == 'PrivateTest']

    def preprocess(pixels: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess an image from the FER2013 dataset.

        Args:
            pixels (tf.Tensor):
                Text string with pixel values separated by spaces.
            label (tf.Tensor):
                Numeric label of the emotion.

        Returns:
            Tuple:
                Preprocessed image and associated label.
        """
        # Conversion of the pixel string to a numerical tensor
        parts = tf.strings.split(pixels, ' ')
        parts = tf.strings.to_number(parts, out_type=tf.float32)

        # Reconstruction of the original image (48x48, 1 channel)
        img = tf.reshape(parts, (48, 48, 1))

        # Conversion to RGB for compatibility with MobileNetV2
        img = tf.image.grayscale_to_rgb(img)

        # Resized to the target size defined for FER
        img = tf.image.resize(img, FER_IMAGE_SIZE, method='bilinear')

        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices(
        tensors=(
            train_df['pixels'].astype(str).values, 
            train_df['emotion'].values.astype('int32')
        )
    )

    valid_ds = tf.data.Dataset.from_tensor_slices(
        tensors=(
            valid_df['pixels'].astype(str).values, 
            valid_df['emotion'].values.astype('int32')
        )
    )

    test_ds = tf.data.Dataset.from_tensor_slices(
        tensors=(
            test_df['pixels'].astype(str).values, 
            test_df['emotion'].values.astype('int32')
        )
    )

    # Shuffling the training dataset to avoid bias
    train_ds = train_ds.shuffle(
        buffer_size=len(train_df), 
        seed=SEED, 
        reshuffle_each_iteration=True
    )

    train_ds = train_ds.map(
        preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    valid_ds = valid_ds.map(
        preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(
        preprocess, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE