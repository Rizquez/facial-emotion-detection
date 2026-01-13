# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import os, argparse
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from src.models import *
from src.loaders import *
from src.webcam.activate import activate_webcam
from common.constants import CK_WEIGHTS_FILE, FER_WEIGHTS_FILE
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

def _train_if_needed(source: str, retrain: bool) -> None:
    """
    # TODO: Documentation
    """
    source = source.lower().strip()

    if source == 'ck':
        if retrain or not os.path.exists(CK_WEIGHTS_FILE):
            train_ds, valid_ds = load_ck_datasets()
            model = build_ck_model()
            train_ck_model(model, train_ds, valid_ds)
        return

    if source == 'fer':
        if retrain or not os.path.exists(FER_WEIGHTS_FILE):
            train_ds, valid_ds, test_ds = load_fer_datasets()
            model = build_fer_model()
            train_fer_model(model, train_ds, valid_ds)
            fine_tune_fer_model(model, train_ds, valid_ds)
            model.evaluate(test_ds, verbose=2)
        return

    raise ValueError('The source parameter must be equal to `ck` or `fer`')

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

    args = parser.parse_args()

    _train_if_needed(args.source, args.retrain)
    activate_webcam(args.source)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE