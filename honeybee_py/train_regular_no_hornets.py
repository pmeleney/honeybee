"""
Train the honeybee regular policy (no hornets) to maximize reward.

- Supervised objective derived from shortest-path targets:
  - If bee.has_food == False → target move is toward nearest flower with food
  - If bee.has_food == True → target move is toward queen
- Optimizer: Adam; Loss: categorical crossentropy
- Observation: 5-dim vector [flower_x, flower_y, queen_x, queen_y, has_food] (positions normalized)
- Output: 4-way move probabilities (up, down, right, left)
- Artifacts: Keras model to `keras_models/regular_model.keras`, CSV weights/biases under `best_weights_and_biases/run_.../`
"""

import os
import datetime
import uuid
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

from .game.game import Game


def build_policy_model(hidden_units: int = 16) -> keras.Model:
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


def get_regular_inputs(game: Game, bee, nearest_flower):
    # Return 5-dim input: [dx_to_flower, dy_to_flower, dx_to_queen, dy_to_queen, has_food]
    beex, beey = bee.position
    flower_x, flower_y = nearest_flower.position[0], nearest_flower.position[1]
    queen_x, queen_y = game.queen.position[0]
    grid = float(game.game_board.shape[0])
    x = np.array([
        (beex - flower_x) / grid,
        (beey - flower_y) / grid,
        (beex - queen_x) / grid,
        (beey - queen_y) / grid,
        float(bee.has_food),
    ], dtype=np.float32)
    return np.expand_dims(x, 0)

def compute_target_move(bee, target_pos) -> int:
    """Return target action index (0 up, 1 down, 2 right, 3 left) towards target_pos."""
    beex, beey = bee.position
    otherx, othery = target_pos
    x_dist = beex - otherx
    y_dist = beey - othery
    if abs(x_dist) >= abs(y_dist):
        return 3 if x_dist > 0 else 2  # left or right
    else:
        return 0 if y_dist > 0 else 1  # up or down


def step_teacher_forced(game: Game):
    """
    Single environment step with supervised targets and teacher-forced movement.
    Returns batched (states, targets) for all bees this step.
    """
    x_batch = []
    y_batch = []

    for bee in game.bees:
        flowers = game.flowers.copy()
        nearest_flower = bee.find_nearest_flower_with_food(flowers)
        obs = get_regular_inputs(game, bee, nearest_flower)[0]

        if bee.has_food:
            # target is queen
            qx, qy = game.queen.position[0]
            target_idx = compute_target_move(bee, (qx, qy))
        else:
            # target is nearest flower with food
            target_idx = compute_target_move(bee, tuple(nearest_flower.position))

        # Teacher-forced movement along target action
        one_hot = np.zeros((1, 4), dtype=np.float32)
        one_hot[0, target_idx] = 1.0
        bee.position = game.net_move(bee, one_hot)

        # Handle overlaps and scoring
        overlap = bee.check_overlap(game.queen, game.flowers, [])
        if overlap == 'Flower' and (np.array(bee.position) == np.array(nearest_flower.position)).all():
            bee.get_food(nearest_flower)
        if overlap == 'Queen' and bee.has_food:
            bee.drop_food()
            bee.score += 1

        x_batch.append(obs)
        target = np.zeros(4, dtype=np.float32)
        target[target_idx] = 1.0
        y_batch.append(target)

    # Update board and time
    game.game_board = game.update_game_board()
    game.game_vars.turn_num += 1

    return np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)


def compute_returns(*_args, **_kwargs):
    # Not used in supervised mode; kept for compatibility.
    return None


def generate_synthetic_dataset(board_size: int = 20, sample_size: int = 200_000):
    """Generate (X, Y) pairs offline using shortest-path rules.
    X: [bee_x, bee_y, flower_x, flower_y, queen_x, queen_y, has_food] normalized to grid
    Y: one-hot 4-way move toward flower if has_food==0 else toward queen
    """
    # Generate raw integer positions for bee, flower, queen; and has_food bit
    raw = np.random.randint(0, board_size, size=(sample_size, 6)).astype(np.float32)
    has_food_bit = np.random.randint(0, 2, size=(sample_size, 1)).astype(np.float32)

    beex, beey = raw[:, 0], raw[:, 1]
    flowerx, flowery = raw[:, 2], raw[:, 3]
    queenx, queeny = raw[:, 4], raw[:, 5]
    has_food = has_food_bit[:, 0] > 0.5

    dx_flower = beex - flowerx
    dy_flower = beey - flowery
    dx_queen = beex - queenx
    dy_queen = beey - queeny

    use_flower = ~has_food
    abs_dx = np.where(use_flower, np.abs(dx_flower), np.abs(dx_queen))
    abs_dy = np.where(use_flower, np.abs(dy_flower), np.abs(dy_queen))
    dx = np.where(use_flower, dx_flower, dx_queen)
    dy = np.where(use_flower, dy_flower, dy_queen)

    move_horizontal = abs_dx >= abs_dy
    action = np.where(
        move_horizontal,
        np.where(dx > 0, 3, 2),
        np.where(dy > 0, 0, 1),
    ).astype(np.int64)

    y = np.zeros((sample_size, 4), dtype=np.float32)
    y[np.arange(sample_size), action] = 1.0
    # Build 5-dim inputs [dxf, dyf, dxq, dyq, has_food], normalized by board size
    X = np.stack(
        [
            (beex - flowerx) / float(board_size),
            (beey - flowery) / float(board_size),
            (beex - queenx) / float(board_size),
            (beey - queeny) / float(board_size),
            has_food_bit[:, 0],
        ],
        axis=1,
    ).astype(np.float32)
    return X, y


def train_regular_policy(
    epochs: int = 100,
    dataset_size: int = 200_000,
    batch_size: int = 1024,
    board_size: int = 20,
    use_mixed_precision: bool = False,
):
    if use_mixed_precision:
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print("[train] Mixed precision enabled (float16)")
        except Exception:
            print("[train] Mixed precision not available; continuing in float32")

    model = build_policy_model(hidden_units=16)
    print("[train] Initialized policy model: 5→16(relu)→16(relu)→4(softmax)")

    X, y = generate_synthetic_dataset(board_size=board_size, sample_size=dataset_size)
    split = int(0.9 * dataset_size)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    class Every5Epochs(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                logs = logs or {}
                print(
                    f"[train] Epoch {epoch+1}/{epochs} - loss={logs.get('loss'):.4f} acc={logs.get('accuracy'):.4f} val_loss={logs.get('val_loss'):.4f} val_acc={logs.get('val_accuracy'):.4f}"
                )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0,
        callbacks=[Every5Epochs()],
    )

    # Save artifacts relative to this file's directory to avoid CWD issues
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, 'keras_models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'regular_model.keras')
    model.save(model_path)
    print(f"[train] Saved Keras model to {model_path}")

    # Note: no CSV export of weights/biases; model is saved as a single Keras file above

    return model


if __name__ == "__main__":
    # Reasonable defaults for a quick, fast training run
    train_regular_policy(epochs=32, dataset_size=200_000, batch_size=1024, board_size=20, use_mixed_precision=False)
