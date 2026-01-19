# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import os, argparse
from typing import Literal
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from src.models import *
from src.loaders import *
from src.utils.webcam import activate_webcam
from src.utils.metrics import offline_evaluation
from common.constants import CK_WEIGHTS_FILE, FER_WEIGHTS_FILE, FER_EMOTION_LABELS
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

def _train_if_needed(source: Literal['ck', 'fer'], retrain: bool, evaluate: bool) -> None:
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
        if retrain or not os.path.exists(CK_WEIGHTS_FILE):
            train_ds, valid_ds = load_ck_datasets()
            model = build_ck_model()
            train_ck_model(model, train_ds, valid_ds)
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
            model.evaluate(test_ds, verbose=2) # Baseline evaluation (loss and accuracy)
            offline_evaluation(
                model,
                test_ds,
                FER_EMOTION_LABELS,
                title='FER2013 Assessment (test) - Extended metrics',
            ) # Extended evaluation (precision/recall/f1 + confusion matrix)
        return

    raise ValueError("The source parameter must be equal to `ck` or `fer`")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Facial emotion detection - Train & Webcam")

    parser.add_argument(
        '--source',
        choices=['ck', 'fer'],
        default='ck',
        help="Source of data on which the model training will be performed (if necessary)"
    )

    parser.add_argument(
        '--retrain',
        action='store_true',
        help="Force retraining even if weights exist"
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help="Perform offline evaluation (F1/recall/precision + confusion matrix) if applicable"
    )

    args = parser.parse_args()

    _train_if_needed(args.source, args.retrain, args.evaluate)
    activate_webcam(args.source)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE