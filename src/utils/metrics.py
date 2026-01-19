# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import List, TYPE_CHECKING
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

__all__ = ['offline_evaluation', 'realtime_performance']

def offline_evaluation(
    model: 'Model',
    dataset: 'tf.data.Dataset',
    labels: List[str],
    *,
    title: str = 'Assessment',
) -> None:
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
    """
    y_true: List[int] = []
    y_pred: List[int] = []

    # We go through the dataset batch by batch to collect actual labels and predictions.
    for x_batch, y_batch in dataset:
        probs = model.predict(x_batch, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds.tolist())

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # Console display - Classification report (precision/recall/f1 per class)
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0 # Avoid warnings if a class does not appear in predictions.
        )
    )
    print(f"\n{'=' * 60}\nConfusion matrix (rows=actual, columns=predicted)\n{'=' * 60}")
    print(cm)

def realtime_performance(frame_times_ms: List[float], *, title: str = 'Real-time performance') -> None:
    """
    Summarizes real-time performance metrics based on frame times.

    Args:
        frame_times_ms (List[float]):
            List of frame durations. Each value must represent the total processing cycle time.
        title (str):
            Title to be displayed in the console.
    """
    if not frame_times_ms:
        print(f"{title}: There is no time data to summarize")
        return

    arr = np.array(frame_times_ms, dtype=np.float64)

    average_latency_ms = float(np.mean(arr))
    latency_p50_ms = float(np.percentile(arr, 50))
    latency_p95_ms = float(np.percentile(arr, 95))
    maximum_latency_ms = float(np.max(arr))

    average_fps = float(1000 / average_latency_ms) if average_latency_ms > 0 else 0.0

    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Average latency (ms): {average_latency_ms:.2f}")
    print(f"P50 latency (ms): {latency_p50_ms:.2f}")
    print(f"P95 latency (ms): {latency_p95_ms:.2f}")
    print(f"Max latency (ms): {maximum_latency_ms:.2f}")

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE