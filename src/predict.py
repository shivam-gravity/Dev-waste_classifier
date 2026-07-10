import argparse

import numpy as np
import tensorflow as tf

from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES


def predict_image(image_path, model=None):
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), dict(zip(CLASS_NAMES, preds.tolist()))


def main():
    parser = argparse.ArgumentParser(description="Classify a single waste image.")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    label, confidence, all_probs = predict_image(args.image_path)
    print(f"Prediction: {label} ({confidence * 100:.1f}% confidence)")
    for cls, prob in sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {cls:<20} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
