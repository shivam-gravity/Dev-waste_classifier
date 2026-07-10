import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import MODEL_PATH, OUTPUT_DIR, CLASS_NAMES
from dataset import get_datasets


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, _, test_ds = get_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f}")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()
