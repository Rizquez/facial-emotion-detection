# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import os, argparse, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencing TensorFlow logs
warnings.filterwarnings('ignore', category=FutureWarning) # Silencing Python future warning
warnings.filterwarnings('ignore', category=UserWarning) # Silencing Python user warning
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

def _train_if_needed(source: str) -> None:
    """
    # TODO: Documentation
    """
    source = source.lower().strip()

    if source == 'ck':
        if not os.path.exists(CK_WEIGHTS_FILE):
            train_ds, valid_ds = load_ck_datasets()
            model = build_ck_model()
            train_ck_model(model, train_ds, valid_ds)
        return

    if source == 'fer':
        if not os.path.exists(FER_WEIGHTS_FILE):
            train_ds, valid_ds = load_fer_datasets()
            model = build_fer_model()
            train_fer_model(model, train_ds, valid_ds)
            fine_tune_fer_model(model, train_ds, valid_ds)
            model.save_weights(FER_WEIGHTS_FILE)
        return

    raise ValueError('The source parameter must be equal to `ck` or `fer`')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Facial emotion detection - Train & Webcam")

    parser.add_argument(
        '--source',
        choices=['ck', 'fer'],
        default='ck',
        help="Model to use: CK+48 (CNN CK+) or FER (MobileNetV2 FER)"
    )

    args = parser.parse_args()

    _train_if_needed(args.source)

    activate_webcam(args.source)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE