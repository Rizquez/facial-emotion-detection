# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from keras.models import Model
from keras import optimizers, losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import FER_IMAGE_SIZE, FER_EMOTION_LABELS, FER_WEIGHTS_FILE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['build_fer_model', 'train_fer_model', 'fine_tune_fer_model']

SHAPE = (FER_IMAGE_SIZE[0], FER_IMAGE_SIZE[1], 3)
"""
# TODO: Documentation
"""

def build_fer_model() -> Model:
    """
    # TODO: Documentation
    """
    inputs = Input(shape=SHAPE)
    preprocessed = preprocess_input(inputs)

    mobilenet_backbone = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=SHAPE,
        name='mobilenetv2_backbone'
    )

    mobilenet_backbone.trainable = False

    features = mobilenet_backbone(preprocessed, training=False)

    pooled = GlobalAveragePooling2D()(features)
    pooled = Dropout(rate=0.3)(pooled)

    dense = Dense(units=256, activation='relu')(pooled)
    dense = Dropout(rate=0.3)(dense)

    outputs = Dense(units=len(FER_EMOTION_LABELS), activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

def train_fer_model(model: Model, train_ds: tf.data.Dataset, valid_ds: tf.data.Dataset) -> None:
    """
    # TODO: Documentation
    """
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=4, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=2, 
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=FER_WEIGHTS_FILE, 
            monitor='val_accuracy', 
            save_weights_only=True, 
            save_best_only=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=12,
        callbacks=callbacks
    )

def fine_tune_fer_model(
    model: Model, 
    train_ds: tf.data.Dataset, 
    valid_ds: tf.data.Dataset, 
    *,
    unfreeze_last: int = 20
) -> None:
    """
    # TODO: Documentation
    """
    mobilenet_backbone  = model.get_layer('mobilenetv2_backbone')

    for layer in mobilenet_backbone.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in mobilenet_backbone.layers[-unfreeze_last:]:
        layer.trainable = True

    for layer in mobilenet_backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=3, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=1, 
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=FER_WEIGHTS_FILE, 
            monitor='val_accuracy', 
            save_weights_only=True, 
            save_best_only=True
        )
    ]

    model.fit(
        train_ds, 
        validation_data=valid_ds, 
        epochs=6, 
        callbacks=callbacks
    )

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE