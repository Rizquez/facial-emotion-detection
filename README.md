# Facial emotion detection

## 🧾 Project description


## 📑 Context


## 🛠️ Key features


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
python main.py --source=fer --retrain 2>nul
```

Where:

- **--source:** Source of data on which the model training will be performed (if necessary).
- **--retrain:** Force retraining even if weights exist.

> [!NOTE]
> For more details about the parameters and execution arguments, see the file: *main.py*

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
├── src
│   ├── loaders
│   │   ├── __init__.py
│   │   ├── ck.py
│   │   └── fer.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn.py                              # CNN for CK+
│   │   └── mobilenetv2.py                      # MobileNetV2 (light backbone) for FER
│   └── webcam
│       └── activate.py
├── .gitattributes
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

## 🎯 Additional considerations for developers

### Forward References (PEP 484)

The project uses *Forward References* according to *PEP 484*. By using `TYPE_CHECKING`, the import of a class is only performed at static type checking time (for example, with *mypy*). During execution, `TYPE_CHECKING` evaluates to `False`, preventing the actual import. This optimizes performance and allows forward references to classes.

Example:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import MyFirstClass

class MySecondClass:
    def do_something(self, first: 'MyFirstClass') -> None:
        pass
```

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