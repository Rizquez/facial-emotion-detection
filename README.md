# Facial emotion detection

## 🧾 Project description

The main objective of the project is to design and implement a system based on convolutional neural networks (CNNs) capable of recognizing human emotions from facial images captured in real time by cameras.

The system allows basic emotions such as happiness, sadness, anger, surprise, fear, disgust, and neutrality to be classified using modern computer vision and deep learning techniques. Special attention is paid to real-time performance, model accuracy, and robustness to variations in lighting and subjects.

## 📑 Context

Automatic facial emotion detection is a key area within artificial intelligence and human-computer interaction, with direct applications in fields such as customer service, security, behavior analysis, and personalized recommendation systems.

Recent advances in deep learning, and in particular in CNNs, have significantly improved the ability of systems to interpret facial expressions with high levels of accuracy. However, many existing approaches have limitations in real-world scenarios, especially in changing lighting conditions or when real-time processing is required.

## 🛠️ Key features

- Real-time facial emotion recognition system based on video captured by camera.
- Classification of 7 basic emotions: happiness, sadness, anger, surprise, fear, disgust, and neutral.
- Model based on convolutional neural networks (CNNs) trained for computer vision tasks.

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
python main.py --source=...
```

Where:
- **source:** Source of data on which the model training will be performed (if necessary).

> [!NOTE]
> For more details about the parameters and execution arguments, see the file: *main.py*

## 📂 Project structure

The main files are organized into:

```
facial-emotion-detection/
├── assets
│   ├── CK+48/...                               # Dataset containing images of human faces grouped by folder
│   ├── FER.csv                                 # Dataset containing grayscale images, size 48×48 pixels
│   └── haarcascade_frontalface_default.xml     # OpenCV Haar Classifier
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

## 📖 Additional documentation

- [CK+ Kaggle](https://www.kaggle.com/code/shawon10/ck-facial-expression-detection)
- [FER Kaggle](https://www.kaggle.com/code/enesztrk/facial-emotion-recognition-vgg19-fer2013)
- [OpenCV Haar Cascades](https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)

## 🔒 License

This project is licensed under the *MIT* license, which allows its use, distribution, and modification under the conditions specified in the *LICENSE* file.

## ⚙ Contact, support, and development

- Pedro Rizquez: pedro.rizquez.94@hotmail.com