"""
Train the hornet-confrontation policy with 5-D inputs to match the game runtime:
Inputs: [dx_to_hornet, dy_to_hornet, dx_to_queen, dy_to_queen, inv_hornet_exists]
Outputs: 4-way move probabilities (up, down, right, left)
Target: shortest-path move TOWARD the hornet (confront)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(hidden_units: int = 16) -> keras.Model:
    inputs = keras.Input(shape=(5,), dtype=tf.float32)
    x = layers.Dense(hidden_units, activation="relu")(inputs)
    x = layers.Dense(hidden_units, activation="relu")(x)
    outputs = layers.Dense(4, activation="softmax", dtype=tf.float32)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_dataset(board_size: int = 20, sample_size: int = 200_000):
    """Synthetic dataset for confrontation policy.
    X: [dxh, dyh, dxq, dyq, inv_exists] normalized by board_size; inv_exists=0 (hornet present)
    y: one-hot 4-way move toward the hornet
    """
    # raw integer positions
    raw = np.random.randint(0, board_size, size=(sample_size, 6)).astype(np.float32)
    beex, beey = raw[:, 0], raw[:, 1]
    hx, hy = raw[:, 2], raw[:, 3]
    qx, qy = raw[:, 4], raw[:, 5]
    inv_exists = np.zeros((sample_size,), dtype=np.float32)  # hornet exists during training

    dx = beex - hx
    dy = beey - hy
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)

    move_horizontal = abs_dx >= abs_dy
    action = np.where(
        move_horizontal,
        np.where(dx > 0, 3, 2),
        np.where(dy > 0, 0, 1),
    ).astype(np.int64)

    y = np.zeros((sample_size, 4), dtype=np.float32)
    y[np.arange(sample_size), action] = 1.0

    X = np.stack(
        [
            dx / float(board_size),
            dy / float(board_size),
            (beex - qx) / float(board_size),
            (beey - qy) / float(board_size),
            inv_exists,
        ],
        axis=1,
    ).astype(np.float32)
    return X, y


def train(epochs: int = 100, dataset_size: int = 200_000, batch_size: int = 1024, board_size: int = 20):
    model = build_model(hidden_units=16)
    print("[train] Hornet model: 5→16(relu)→16(relu)→4(softmax)")

    X, y = generate_dataset(board_size=board_size, sample_size=dataset_size)
    split = int(0.9 * dataset_size)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    class Every5Epochs(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                logs = logs or {}
                print(
                    f"[train] Epoch {epoch+1}/{epochs} - loss={logs.get('loss'):.4f} acc={logs.get('accuracy'):.4f} val_loss={logs.get('val_loss'):.4f} val_acc={logs.get('val_accuracy'):.4f}"
                )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0, callbacks=[Every5Epochs()])

    os.makedirs('keras_models', exist_ok=True)
    out_path = os.path.join('keras_models', 'hornet_model.keras')
    model.save(out_path)
    print(f"[train] Saved hornet model to {out_path}")

    # CSV export removed: keep only the Keras model

    return model


if __name__ == "__main__":
    train(epochs=100, dataset_size=200_000, batch_size=1024, board_size=20)
