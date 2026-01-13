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
    # TODO: Documentation
    """
    df = pd.read_csv(FER_FILE)

    train_df = df[df['Usage'] == 'Training']
    valid_df = df[df['Usage'] == 'PublicTest']
    test_df  = df[df['Usage'] == 'PrivateTest']

    def preprocess(pixels, label):
        """
        # TODO: Documentation
        """
        parts = tf.strings.split(pixels, ' ')
        parts = tf.strings.to_number(parts, out_type=tf.float32)

        img = tf.reshape(parts, (48, 48, 1))
        img = tf.image.grayscale_to_rgb(img)
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