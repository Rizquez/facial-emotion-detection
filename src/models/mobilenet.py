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
Model input format for the FER2013 dataset.

Although FER2013 is originally grayscale (48x48), the images are converted to RGB and resized to be compatible
with pre-trained models such as MobileNetV2.
"""

def build_fer_model() -> Model:
    """
    Build and compile the emotion classification model based on `MobileNetV2`.

    This model applies transfer learning using pre-trained `MobileNetV2` on `ImageNet` as a feature extractor, 
    adding dense final layers for facial emotion classification.

    **Key features:**
        - Frozen pre-trained backbone (initial phase).
        - Global pooling to reduce dimensionality.
        - Dense layers with Dropout to avoid overfitting.
        - Optimized for CPU training.

    Returns:
        Model:
            Compiled model ready for training or inference.
    """
    # Definition of the model input
    inputs = Input(shape=SHAPE)

    # Specific preprocessing required by MobileNetV2
    preprocessed = preprocess_input(inputs)

    # Pre-trained MobileNetV2 backbone on ImageNet
    mobilenet_backbone = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=SHAPE,
        name='mobilenetv2_backbone'
    )

    # We freeze the backbone for the initial training phase.
    mobilenet_backbone.trainable = False

    # Feature extraction (inference in non-trainable mode)
    features = mobilenet_backbone(preprocessed, training=False)

    # Global pooling to reduce the number of parameters
    pooled = GlobalAveragePooling2D()(features)
    pooled = Dropout(rate=0.3)(pooled)

    # Final dense layers for classification
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
    Train the FER model using transfer learning (frozen backbone).

    **This function:**
        - Trains only the added top layers.
        - Uses callbacks to avoid overfitting.
        - Automatically saves the best weights based on validation accuracy.

    Args:
        model (Model):
            Previously built and compiled model.
        train_ds (tf.data.Dataset):
            Training dataset.
        valid_ds (tf.data.Dataset):
            Validation dataset.
    """
    callbacks = [
        # Stop training if there is no improvement in validation
        EarlyStopping(
            monitor='val_accuracy', 
            patience=4, 
            restore_best_weights=True
        ),
        # Reduce the learning rate if the loss stagnates
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=2, 
            min_lr=1e-6
        ),
        # Save the best weights from the model
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
    Perform fine-tuning of the FER model by unfreezing the last layers of the backbone.

    **This function:**
        - The last convolutional layers of MobileNetV2 are unfrozen.
        - BatchNormalization is kept frozen for stability.
        - The learning rate is reduced for fine adjustments.
        - The model's adaptation to the facial emotion domain is improved.

    Args:
        model (Model):
            Previously trained model.
        train_ds (tf.data.Dataset):
            Training dataset.
        valid_ds (tf.data.Dataset):
            Validation dataset.
        unfreeze_last (int, optional):
            Number of final layers of the backbone that are unfrozen.
    """
    # Access to the MobileNetV2 backbone by name
    mobilenet_backbone  = model.get_layer('mobilenetv2_backbone')

    # We freeze all layers except the last ones 'unfreeze_last'
    for layer in mobilenet_backbone.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in mobilenet_backbone.layers[-unfreeze_last:]:
        layer.trainable = True

    # BatchNormalization layers are kept frozen to prevent instability.
    for layer in mobilenet_backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    # Recompilation of the model with a lower learning rate
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