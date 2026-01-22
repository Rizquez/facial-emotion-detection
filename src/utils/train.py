# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import os
from typing import Literal
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from src.models import *
from src.loaders import *
from src.utils.metrics import offline_evaluation
from common.constants import CK_WEIGHTS_FILE, FER_WEIGHTS_FILE, FER_EMOTION_LABELS, CK_EMOTION_LABELS
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

def train_if_needed(source: Literal['ck', 'fer'], retrain: bool, evaluate: bool) -> None:
    """
    Train the corresponding model if necessary, according to the indicated source.

    Args:
        source (Literal['ck', 'fer'], optional):
            Indicates the pipeline to use: ck for CNN trained with CK+ and fer for MobileNetV2 trained with FER2013.
        retrain (bool):
            If True, forces retraining even if weight files exist.
        evaluate (bool):
            If True, performs the extended evaluation on FER.
    
    Raises:
        ValueError:
            If `source` is neither ck nor fer.
    """
    source = source.lower().strip()

    if source == 'ck':
        train_ds, valid_ds = load_ck_datasets()
        model = build_ck_model()

        if retrain or not os.path.exists(CK_WEIGHTS_FILE):
            train_ck_model(model, train_ds, valid_ds)
        else:
            model.load_weights(CK_WEIGHTS_FILE)

        if evaluate:
            model.evaluate(valid_ds, verbose=2)
            offline_evaluation(
                model,
                valid_ds,
                CK_EMOTION_LABELS,
                title='CK+ Assessment (validation) - Extended metrics'
            )
        return

    if source == 'fer':
        need_train = retrain or not os.path.exists(FER_WEIGHTS_FILE)

        train_ds, valid_ds, test_ds = load_fer_datasets()
        model = build_fer_model()

        if need_train:
            train_fer_model(model, train_ds, valid_ds)
            fine_tune_fer_model(model, train_ds, valid_ds)
        else:
            model.load_weights(FER_WEIGHTS_FILE)

        if evaluate:
            model.evaluate(test_ds, verbose=2)
            offline_evaluation(
                model,
                test_ds,
                FER_EMOTION_LABELS,
                title='FER2013 Assessment (test) - Extended metrics'
            )
        return

    raise ValueError("The source parameter must be equal to `ck` or `fer`")

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE