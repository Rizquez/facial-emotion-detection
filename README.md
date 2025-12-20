# Facial emotion detection

## 🧾 Project description

...

## 📑 Context

...

## 🛠️ Key features

...

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

### Console

To run the application from the console, you can use the following command:

```sh
python main.py
```

## 📂 Project structure

The main files are organized into:

```
facial-emotion-detection/
├── CK+48/...                   # Data set containing images of human faces grouped by folder
├── common
│   └── constants.py
├── src
│   ├── models
│   │   └── cnn.py
│   └── utils
│       └── loaders.py
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

- [CK+ Kaggle](https://www.kaggle.com/datasets/shawon10/ckplus/data)

## 🔒 License

This project is licensed under the *MIT* license, which allows its use, distribution, and modification under the conditions specified in the *LICENSE* file.

## ⚙ Contact, support, and development

- Pedro Rizquez: pedro.rizquez.94@hotmail.com