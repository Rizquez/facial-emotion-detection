# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
from argparse import ArgumentParser

if TYPE_CHECKING:
    from argparse import Namespace
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
# Get listed here!
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['obtain_args']

def obtain_args() -> 'Namespace':
    """
    Defines and processes console arguments for algorithm execution.

    Returns:
        Namespace: 
            Object with parsed and validated arguments.
    """
    parser = ArgumentParser(description="Facial emotion detection - Train & Webcam")

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

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help="Measures real-time performance (FPS/latency) during webcam execution"
    )

    parser.add_argument(
        '--seconds',
        type=int,
        default=30,
        help="Duration (in seconds) of the real-time benchmark, default 30 seconds"
    )

    return parser.parse_args()

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE