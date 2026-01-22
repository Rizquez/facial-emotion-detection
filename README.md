# Facial emotion detection

## 🧾 Project description

This project implements a comprehensive real-time facial emotion detection system using deep learning and computer vision techniques.
The system is capable of training emotion classification models from different datasets and then applying those models to images captured from a webcam, detecting faces and classifying the predominant emotion.

The solution combines convolutional neural networks (CNN) for learning facial features with an efficient face detector, enabling a complete workflow from data preparation and model training to real-time inference.

The project is designed with a modular and extensible architecture, facilitating experimentation with different models, datasets, and configurations.

## 📑 Context

Automatic facial emotion detection is an important area within Artificial Intelligence, with applications in fields such as human-machine interaction, behavior analysis, education, health, video games, and recommendation systems.

This project is a practical implementation to explore:

- The training of emotion classification models based on facial images.
- The comparison between classic CNN-based models and lighter, efficiency-oriented architectures.
- The integration of trained models with real-time video capture systems.

Due to common limitations in development environments (such as the absence of CUDA-compatible GPUs on Windows), optimized architectures were chosen that allow the system to be trained and run reasonably on CPUs, without sacrificing pipeline consistency or the overall quality of the approach.

## 🛠️ Key features

- Training facial emotion detection models using different datasets.
- Support for multiple data sources (`CK+` and `FER2013`).
- Implementation of:
    - Custom CNN for grayscale images.
    - Model based on MobileNetV2 as a lightweight backbone.
- Real-time face detection using MediaPipe.
- Live emotion classification using the webcam.
- Automatic image normalization and preprocessing.
- Use of data augmentation techniques to improve model generalization.
- Complete pipeline:
    - Data loading
    - Training
    - Fine-tuning
    - Evaluation
    - Real-time inference
- Modular and easy-to-extend architecture.
- Configurable execution from the console using arguments.

## 📊 Results

### 🔹 Results with CK+ (custom CNN)

The model trained on the **CK+** dataset uses a convolutional neural network designed specifically for small grayscale images (48×48).

- **Number of classes:** 7 emotions.
- **Dataset size:**
    - Training: 785 images.
    - Validation: 196 images.
- **Training epochs:** 25.

**Observed performance:**

- Training accuracy remains around **23–25%** at the end of training.
- Validation accuracy reaches values close to **24%**, with slight fluctuations between epochs.
- The loss function shows a progressive reduction, indicating that the model learns basic patterns, albeit with limited generalization capacity.

**Interpretation:**

The relatively low performance is mainly due to:

- The small size of the dataset.
- The high variability between subjects and expressions.
- Training performed exclusively on CPU.
- The inherent complexity of classifying facial emotions with few samples.

This model is mainly used as an **experimental baseline** and as support for integration into the complete real-time detection pipeline.

### 🔹 Results with FER2013 (MobileNetV2)

For the **FER2013** dataset, an architecture based on **MobileNetV2** pre-trained on ImageNet was used, followed by a partial fine-tuning phase.

- **Number of classes:** 7 emotions.
- **Images:** 48×48 grayscale, resized to 96×96 and converted to RGB.
- **Initial training:** 12 epochs.
- **Fine-tuning:** 6 epochs (unfreezing final layers).

**Initial training**

- Final training accuracy: **~52%**
- Validation accuracy: **~51–52%**
- Loss decreases steadily, showing consistent improvement between epochs.

**Fine-tuning**

- Final training accuracy: **~60%**
- Final validation accuracy: **~55–56%**
- The fine-tuning phase improves the model's ability to adapt to the specific domain of facial emotions.

**Interpretation:**

The use of a lightweight pre-trained backbone allows us to:

- Accelerate training in environments without GPUs.
- Obtain reasonable performance despite the noise present in the FER2013 labels.
- Maintain a balance between accuracy and computational efficiency.

This model is the main one used for real-time emotion detection via webcam.

## ⚠️ Limitations

- Emotion recognition is limited to the classes available in the datasets.
- Performance may decrease in low-light conditions, with extreme head postures, if the subject has a beard, or is wearing a hat.
- The models were trained without GPU acceleration, which limits experimentation with larger architectures.
- FER2013 labels may contain noise due to automatic annotation.

## 💽 Installation (Windows)

Clone this repository (ssh):
```sh
git clone git@github.com:Rizquez/facial-emotion-detection.git
```

Access the project directory:
```sh
cd facial-emotion-detection
```

Create a development environment using the **virtualenv** library:
```sh
virtualenv venv
```

If you do not have the library installed, you can run:
```sh
python -m venv env
```

Activate the development environment:
```sh
venv\Scripts\activate
```

Once the environment is activated, install the dependencies:
```sh
pip install -r requirements.txt
```

## 🚀 Execution

### Operating requirements

This project uses TensorFlow and Deep Learning libraries, so it is essential to use a compatible version of Python. 

Recent versions such as Python `3.12.x`, `3.13.x`, and `3.14.x` are not currently compatible with TensorFlow. This is because many libraries in the ML ecosystem are 2 or 3 versions behind Python.

> [!NOTE]
> This project was developed using version 3.11.9, which is stable for TensorFlow, Keras, and NumPy.

### Running through the console

To run the application from the console, you can use the following command shown as an example:
```sh
python main.py --source=fer --evaluate --benchmark --seconds=30 --retrain
```

Where:

- **--source:** Source of data on which the model training will be performed (if necessary).
- **--evaluate:** Perform offline evaluation (F1/recall/precision + confusion matrix) if applicable.
- **--benchmark:** Measures real-time performance (FPS/latency) during webcam execution.
- **--seconds:** Duration (in seconds) of the real-time benchmark, default 30 seconds.
- **--retrain:** Force retraining even if weights exist.

> [!NOTE]
> All execution parameters are optional and have default values assigned to them. For more details on the parameters and execution arguments, see the file: *handlers/console.py*

## 📂 Project structure

The main files are organized into:

```
facial-emotion-detection/
├── assets
│   ├── CK+48/...                               # Dataset containing images of human faces grouped by folder
│   ├── blaze_face_short_range.tflite           # A lightweight model for detecting one or more faces in selfie-like images from a smartphone camera or webcam
│   └── FER.csv                                 # Dataset containing grayscale images, size 48×48 pixels
├── common
│   └── constants.py
├── handlers
│   └── console.py
├── helpers
│   └── stdout.py
├── src
│   ├── loaders
│   │   ├── __init__.py
│   │   ├── ck.py
│   │   └── fer.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn.py                              # CNN for CK+
│   │   └── mobilenetv2.py                      # MobileNetV2 (light backbone) for FER
│   └── utils
│       ├── metrics.py
│       ├── train.py
│       └── webcam.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

## 🎯 Additional considerations for developers

### Forward References and Modern Typing (PEP 484 / PEP 563)

The project uses *Forward References* following modern Python recommendations for static typing.

To do this, two complementary mechanisms are combined:

- `from __future__ import annotations`, which delays the evaluation of all type annotations.
- `TYPE_CHECKING`, which allows imports to be performed only during static analysis (for example, with *mypy* or editors such as VS Code), avoiding runtime dependencies.

Example:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import MyFirstClass

class MySecondClass:
    def do_something(self, first: MyFirstClass) -> None:
        pass
```

This combination allows you to:

- Avoid import cycles.
- Not execute unnecessary imports at runtime.
- Write clean and readable type annotations without having to use manual strings.
- Maintain compatibility with static analysis tools.

During normal program execution, `TYPE_CHECKING` evaluates to `False`, so protected imports are not performed. Type annotations are not evaluated at runtime thanks to `from __future__ import annotations`.

### FER (CPU vs GPU)

The *Kaggle notebook* used as a reference for *FER2013* uses *VGG19*. However, in my environment, the GPU is not compatible with *CUDA*, and `TensorFlow (>= 2.11)` no longer offers native GPU support on Windows, so to use the GPU, it would be necessary to work through WSL2 (or other alternatives). Due to these limitations and to avoid excessive training times on the CPU, I replaced VGG19 with a lighter model (`MobileNetV2`) while maintaining the pipeline with *CNN + CK+48 + FER*.

For more information on compatible GPUs and CUDA/cuDNN requirements, check the official NVIDIA documentation (Compute Capability / CUDA on WSL).

## 📖 Additional documentation

- [Kaggle - CK+](https://www.kaggle.com/code/shawon10/ck-facial-expression-detection)
- [Kaggle - FER](https://www.kaggle.com/code/enesztrk/facial-emotion-recognition-vgg19-fer2013)
- [Google - Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector?hl=es-419)
- [CUDA - GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)

## 🔒 License

This project is licensed under the *MIT* license, which allows its use, distribution, and modification under the conditions specified in the *LICENSE* file.

## ⚙ Contact, support, and development

- Pedro Rizquez: pedro.rizquez.94@hotmail.com
