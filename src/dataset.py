import tensorflow as tf
from tensorflow.keras import layers

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES

AUTOTUNE = tf.data.AUTOTUNE

_augment = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ],
    name="augmentation",
)

_rescale = layers.Rescaling(1.0 / 255)


def _load_split(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def get_datasets(augment_train=True):
    """Returns (train_ds, val_ds, test_ds), all normalized to [0, 1].

    Directory layout expected under each of train/val/test:
        <split>/<ClassName>/*.jpg
    Class names must match config.CLASS_NAMES.
    """
    train_ds = _load_split(TRAIN_DIR, shuffle=True)
    val_ds = _load_split(VAL_DIR, shuffle=False)
    test_ds = _load_split(TEST_DIR, shuffle=False)

    def prep(ds, training):
        ds = ds.map(lambda x, y: (_rescale(x), y), num_parallel_calls=AUTOTUNE)
        if training and augment_train:
            ds = ds.map(lambda x, y: (_augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        return ds.prefetch(AUTOTUNE)

    train_ds = prep(train_ds, training=True)
    val_ds = prep(val_ds, training=False)
    test_ds = prep(test_ds, training=False)

    return train_ds, val_ds, test_ds
