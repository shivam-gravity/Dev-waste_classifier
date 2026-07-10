import time

import cv2
import numpy as np
import tensorflow as tf

from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES

CONFIDENCE_THRESHOLD = 0.5


def preprocess_frame(frame):
    resized = cv2.resize(frame, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def main(camera_index=0):
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.time()
            batch = preprocess_frame(frame)
            preds = model.predict(batch, verbose=0)[0]
            elapsed_ms = (time.time() - start) * 1000

            idx = int(np.argmax(preds))
            label, confidence = CLASS_NAMES[idx], float(preds[idx])

            if confidence >= CONFIDENCE_THRESHOLD:
                text = f"{label}: {confidence * 100:.1f}%"
            else:
                text = "Uncertain"

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{elapsed_ms:.0f} ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Waste Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
