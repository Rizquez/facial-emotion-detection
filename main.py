# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencing TensorFlow logs

# Silencing Python warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from src.models.cnn import *
from src.tools.webcam import *
from src.utils.loaders import *
from common.constants import MODEL_PATH
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    if not os.path.exists(MODEL_PATH):
        training, validation = load_datasets()
        model = build_model()
        train_model(model, training, validation)

    run_webcam()

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE