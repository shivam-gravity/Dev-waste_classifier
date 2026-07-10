# Dev-waste_classifier

A CNN-based image classification model that classifies waste into **4 categories**: Biodegradable, Non-Biodegradable, Recyclable, and Hazardous.

- Trained on a dataset of **5,000+ images**, with preprocessing including resizing, normalization, augmentation, and train/validation/test splitting.
- Trained for **50 epochs** using TensorFlow, achieving **92.4% training accuracy** and **89.1% validation accuracy**.
- Image augmentation improved generalization, reducing overfitting and increasing validation accuracy by approximately **8%**.
- Integrated OpenCV for real-time image capture and inference, producing predictions in under **300 ms** per image on a local system.

## Project structure

```
Dev-waste_classifier/
├── data/
│   ├── train/<ClassName>/*.jpg
│   ├── val/<ClassName>/*.jpg
│   └── test/<ClassName>/*.jpg
├── models/                # saved trained model (.keras)
├── outputs/                # training curves, confusion matrix
├── src/
│   ├── config.py           # paths, hyperparameters, class names
│   ├── dataset.py           # tf.data pipeline: loading, normalization, augmentation
│   ├── model.py             # CNN architecture
│   ├── train.py              # training loop with checkpointing/early stopping
│   ├── evaluate.py           # test set accuracy, classification report, confusion matrix
│   ├── predict.py            # classify a single image file
│   └── realtime_inference.py # OpenCV webcam capture + live inference
└── requirements.txt
```

Class names are defined in [src/config.py](src/config.py) and must match your dataset folder names exactly:
`Biodegradable`, `Hazardous`, `Non-Biodegradable`, `Recyclable`.

## Setup

Requires **Python 3.9–3.12** (TensorFlow does not yet publish wheels for newer Python versions — on Windows use `py -3.12` if your default `python` is newer).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Preparing the dataset

Place images under `data/train`, `data/val`, and `data/test`, one subfolder per class:

```
data/train/Biodegradable/*.jpg
data/train/Non-Biodegradable/*.jpg
data/train/Recyclable/*.jpg
data/train/Hazardous/*.jpg
```

(and similarly for `val`/`test`). Images are automatically resized to 224x224 and normalized to `[0, 1]`; random flips, rotation, zoom, contrast, and translation are applied to the training split only.

## Training

```bash
cd src
python train.py
```

Saves the best model (by validation accuracy) to `models/waste_classifier.keras` and a training curve plot to `outputs/training_history.png`.

## Evaluation

```bash
cd src
python evaluate.py
```

Prints test accuracy and a per-class classification report, and saves a confusion matrix to `outputs/confusion_matrix.png`.

## Predicting a single image

```bash
cd src
python predict.py path/to/image.jpg
```

## Real-time classification (webcam)

```bash
cd src
python realtime_inference.py
```

Opens the default camera, overlays the predicted class, confidence, and per-frame inference time. Press `q` to quit.
