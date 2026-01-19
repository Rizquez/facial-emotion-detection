# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import os, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencing TensorFlow logs
warnings.filterwarnings('ignore', category=FutureWarning) # Silencing Python future warning
warnings.filterwarnings('ignore', category=UserWarning) # Silencing Python user warning
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from handlers.console import obtain_args
from src.utils.train import train_if_needed
from src.utils.webcam import activate_webcam
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    args = obtain_args()
    train_if_needed(args.source, args.retrain, args.evaluate)
    activate_webcam(args.source, args.benchmark)

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE