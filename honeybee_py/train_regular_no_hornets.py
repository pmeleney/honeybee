"""
Train the honeybee regular policy (no hornets) to maximize reward.

- Uses on-policy REINFORCE with a simple baseline.
- Environment: Game with hornets disabled.
- Observation: 7-dim vector from `Game.get_regular_inputs(..., input_type='positions')`.
- Action: 4-way move sampled from policy softmax.
- Reward: +1 when food is delivered to the queen; 0 otherwise.
- Outputs: Saves Keras model to `keras_models/regular_model.keras` and CSV weights/biases to `best_weights_and_biases/`.
"""

import os
import datetime
import uuid
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .game.game import Game


def build_policy_model() -> keras.Model:
    inputs = keras.Input(shape=(7,))
    x = layers.Dense(10, activation="relu")(inputs)
    x = layers.Dense(10, activation="sigmoid")(x)
    x = layers.Dense(10, activation="sigmoid")(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3), loss="categorical_crossentropy")
    return model


def sample_action(probabilities: np.ndarray) -> int:
    if probabilities.ndim > 1:
        probabilities = probabilities[0]
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(4, p=probabilities))


def get_regular_inputs(game: Game, bee, nearest_flower):
    return game.get_regular_inputs(bee, nearest_flower, game.queen, input_type="positions")


def step_regular(game: Game, model: keras.Model):
    """
    Single environment step for all bees using only the regular policy.
    Returns the sum of rewards from all bees this step and a list of (state, action, reward) per bee.
    """
    trajectories = []
    step_reward = 0.0

    for bee in game.bees:
        flowers = game.flowers.copy()
        nearest_flower = bee.find_nearest_flower_with_food(flowers)
        obs = get_regular_inputs(game, bee, nearest_flower)
        probs = model.predict(obs, verbose=False)
        action = sample_action(probs)

        # Move bee using chosen action with edge correction via net_move
        one_hot = np.zeros((1, 4), dtype=np.float32)
        one_hot[0, action] = 1.0
        prev_score = bee.score
        bee.position = game.net_move(bee, one_hot)

        # Handle overlaps and scoring (no hornets in this training)
        overlap = bee.check_overlap(game.queen, game.flowers, [])
        if overlap == 'Flower' and (np.array(bee.position) == np.array(nearest_flower.position)).all():
            bee.get_food(nearest_flower)
        if overlap == 'Queen':
            if bee.has_food:
                bee.drop_food()
                bee.score += 1

        reward = float(bee.score - prev_score)
        step_reward += reward
        trajectories.append((obs[0], action, reward))

    # Update board and time
    game.game_board = game.update_game_board()
    game.game_vars.turn_num += 1

    return step_reward, trajectories


def compute_returns(rewards, gamma: float = 0.99):
    g = 0.0
    returns = []
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    returns = np.array(returns, dtype=np.float32)
    # Normalize for stability
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train_regular_policy(
    episodes: int = 200,
    max_steps_per_episode: int = 200,
    gamma: float = 0.99,
    visualize: bool = False,
):
    # Build policy
    model = build_policy_model()
    print("[train] Initialized policy model: 7→10(relu)→10(sigmoid)→10(sigmoid)→4(softmax)")

    # Training loop
    for ep in range(episodes):
        game = Game()
        # Disable hornets
        game.game_state.HORNETS_EXIST = False
        game.hornets = []
        game.hornet_exists = False
        game.game_board = game.update_game_board()
        if (ep % 10) == 0:
            print(f"[train] Episode {ep+1}/{episodes} - starting")

        ep_states = []
        ep_actions = []
        ep_rewards = []

        total_reward = 0.0
        for _ in range(max_steps_per_episode):
            step_reward, traj = step_regular(game, model)
            total_reward += step_reward
            for s, a, r in traj:
                ep_states.append(s)
                ep_actions.append(a)
                ep_rewards.append(r)
            if visualize:
                # optional: slow visualization can be added here
                pass

        returns = compute_returns(ep_rewards, gamma)

        # Prepare supervised-like targets with advantages as sample weights
        x = np.array(ep_states, dtype=np.float32)
        y = np.zeros((len(ep_actions), 4), dtype=np.float32)
        y[np.arange(len(ep_actions)), ep_actions] = 1.0
        w = returns

        model.train_on_batch(x, y, sample_weight=w)
        if (ep + 1) % 5 == 0:
            avg_return = float(np.mean(w)) if len(w) > 0 else 0.0
            print(f"[train] Episode {ep+1}/{episodes} - steps={game.game_vars.turn_num} total_reward={total_reward:.2f} avg_advantage={avg_return:.3f}")

    # Save artifacts
    os.makedirs('keras_models', exist_ok=True)
    model_path = os.path.join('keras_models', 'regular_model.keras')
    model.save(model_path)
    print(f"[train] Saved Keras model to {model_path}")

    # Also export CSV weights/biases to best_weights_and_biases/
    os.makedirs('best_weights_and_biases', exist_ok=True)
    # Create a run-specific subdirectory with timestamp and short id
    run_tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_' + uuid.uuid4().hex[:8]
    run_dir = os.path.join('best_weights_and_biases', f'run_{run_tag}')
    os.makedirs(run_dir, exist_ok=True)
    for i, layer in enumerate(model.layers):
        params = layer.get_weights()
        if len(params) == 2:
            w, b = params
            # Save into run-specific directory
            weights_path = os.path.join(run_dir, f'Best_weights_model_regular_layer_{i}.csv')
            biases_path = os.path.join(run_dir, f'Best_biases_model_regular_layer_{i}.csv')
            np.savetxt(weights_path, w, delimiter=',')
            np.savetxt(biases_path, b, delimiter=',')
            print(f"[train] Saved layer {i} weights to {weights_path}")
            print(f"[train] Saved layer {i} biases to {biases_path}")
            # And also update root files for backward compatibility
            np.savetxt(os.path.join('best_weights_and_biases', f'Best_weights_model_regular_layer_{i}.csv'), w, delimiter=',')
            np.savetxt(os.path.join('best_weights_and_biases', f'Best_biases_model_regular_layer_{i}.csv'), b, delimiter=',')

    return model


if __name__ == "__main__":
    # Reasonable defaults for a quick training run
    train_regular_policy(episodes=200, max_steps_per_episode=200, gamma=0.99, visualize=False)
