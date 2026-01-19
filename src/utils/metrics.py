# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import List, Tuple, Dict, TYPE_CHECKING
from sklearn.metrics import classification_report, confusion_matrix

if TYPE_CHECKING:
    import tensorflow as tf
    from keras import Model
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
# Get listed here!
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['offline_evaluation']

def offline_evaluation(
    model: 'Model',
    dataset: 'tf.data.Dataset',
    labels: List[str],
    *,
    title: str = 'Assessment',
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluates a multi-class classifier and generates standard metrics.

    **This function calculates:**
        - Classification report (precision, recall, f1-score) per class and averages.
        - Confusion matrix.

    **Important:**
        - The evaluation is *offline* (not in real time).
        - The `dataset` must output batches with structure, where `y_batch` are integer indices.

    Args:
        model (Model):
            Model already trained.
        dataset (tf.data.Dataset):
            Dataset to be evaluated.
        labels (List[str]):
            List of class names in the same order as the model output.
        title (str, optional):
            Title to display in the console.

    Returns:
        Tuple:
            - report_dict: Classification report in dict format.
            - cm: Confusion matrix with shape.
    """
    y_true: List[int] = []
    y_pred: List[int] = []

    # We go through the dataset batch by batch to collect actual labels and predictions.
    for x_batch, y_batch in dataset:
        probs = model.predict(x_batch, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds.tolist())

    # Classification report (precision/recall/f1 per class)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,  # Avoid warnings if a class does not appear in predictions.
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # Console display
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    return report_dict, cm

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE