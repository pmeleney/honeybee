from __future__ import annotations

from typing import Any, List, Tuple, Dict
import os
import numpy as np

from .policies.models import load_regular_model, load_hornet_model
from .game.game import Game, visualize, play_game

__all__ = [
    # Back-compat exports for CLI
    "Config",
    "Network",
    "demo",
    "run",
    "write_se_config",
    # Single-neuron API
    "SingleNeuronMoralNet",
    "evolve_moral_single_neuron",
]


# Multi-neuron combined-head path removed; single-neuron moral gate only.


    


    


if __name__ == "__main__":
    # Quick manual run: evolve single-neuron moral gate and print final score
    model = evolve_moral_single_neuron(generations=10, trials_per_gen=8, target_score=10, viz=False)
    print("[moral-1] Evolution finished")


class Config:
    """Minimal config loader to satisfy CLI expectations.

    Backward-compat removed: config files are no longer used.
    """

    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.name = 'morality_layer'


class Network:
    """Placeholder moral network to satisfy demo/evolve call signatures."""

    def __init__(self, _config: Config | None = None) -> None:
        pass

    def run_once(self, _input: bool) -> np.ndarray:
        # Placeholder; will be replaced by SingleNeuronMoralNet below via alias
        return np.array([[0.0]], dtype=np.float32)


def write_se_config() -> None:
    """Deprecated: config files are no longer written or used."""
    return None


def demo(_moral_net: Network, viz: bool = True) -> Tuple[bool, int]:
    """Run a quick demo using a single-neuron moral gate.

    Returns (queen_alive, hive_score).
    """
    reg = load_regular_model()
    horn = load_hornet_model()
    g = Game()
    moral = SingleNeuronMoralNet()
    qa, score, _, _ = play_game(g, moral, reg, horn, viz=viz)
    print({'queen_alive': qa, 'score': score, 'bees_alive': len(g.bees), 'hornets_killed': g.game_vars.hornets_killed})
    return qa, score


def run(_moral_config: Config, viz: bool = True) -> None:
    """Run evolution of the single-neuron moral gate (CLI entry)."""
    # Use defaults (generations=64, trials_per_gen=32) unless overridden here
    _ = evolve_moral_single_neuron(viz=viz)
    return None


# ======================================
# Single-neuron moral gate and evolution
# ======================================

class SingleNeuronMoralNet:
    """A single sigmoid neuron gating between hornet vs regular behaviors.

    Input: hornet_exists (bool → 0.0/1.0)
    Output: scalar y in (0,1). If y > 0.5 → use hornet policy; else regular policy.
    """

    def __init__(self, weight: float | None = None, bias: float | None = None) -> None:
        # Initialize near 0 so sigmoid output starts ~0.5; stdev 0.1
        self.weight = float(np.random.normal(0.0, 0.1)) if weight is None else float(weight)
        self.bias = float(np.random.normal(0.0, 0.1)) if bias is None else float(bias)

    def run_once(self, hornet_exists: bool) -> np.ndarray:
        x = 1.0 if hornet_exists else 0.0
        z = self.weight * x + self.bias
        y = 1.0 / (1.0 + np.exp(-z))
        return np.array([[y]], dtype=np.float32)

    def get_params(self) -> Tuple[float, float]:
        return self.weight, self.bias

    def set_params(self, weight: float, bias: float) -> None:
        self.weight = float(weight)
        self.bias = float(bias)


def evolve_moral_single_neuron(
    generations: int = 8,
    trials_per_gen: int = 16,
    noise_scale: float = 0.10,
    target_score: int = 20,
    viz: bool = False,
) -> SingleNeuronMoralNet:
    """Hill-climb the single-neuron moral gate on hive score with survival priority."""
    reg = load_regular_model()
    horn = load_hornet_model()
    moral = SingleNeuronMoralNet()

    # Track average morality neuron value (for hornet_exists=True) per generation
    moral_values_per_generation: List[float] = []

    def evaluate() -> Tuple[bool, int]:
        g = Game()
        qa, score, _, _ = play_game(g, moral, reg, horn, viz=viz)
        return qa, score

    best_w, best_b = moral.get_params()
    best_alive, best_score = evaluate()
    print(f"[moral-1] init alive={best_alive} score={best_score}")

    # Parents for reproduction in next generation: only alive hives, with scores
    parents: List[Tuple[float, float, int]] = [(best_w, best_b, best_score)] if best_alive else []
    # Previous generation mean morality neuron value (hornet_exists=True)
    prev_mu: float = 0.5

    # Track per-trial morality value time series for plotting
    candidate_series: List[List[float]] = [[] for _ in range(trials_per_gen)]
    candidate_alive: List[List[bool]] = [[] for _ in range(trials_per_gen)]

    for gen in range(generations):
        improved = False
        cand_vals: List[float] = []
        cand_scores_alive: List[int] = []
        # Next generation parent pool (only survivors), with scores for weighted reproduction
        next_parents: List[Tuple[float, float, int]] = []
        # Compute a fallback mu from previous generation's survivors (used only if no parents)
        if parents:
            mu_prev = float(np.mean([1.0 / (1.0 + np.exp(-(pw + pb))) for pw, pb, _sc in parents]))
        else:
            mu_prev = prev_mu
        for trial_idx in range(trials_per_gen):
            # Select a parent among survivors with probability proportional to score
            if parents:
                scores = np.array([max(0, sc) for (_w, _b, sc) in parents], dtype=float)
                # Avoid zero total weight by adding small epsilon
                if scores.sum() <= 0:
                    probs = None  # uniform fallback
                else:
                    probs = scores / scores.sum()
                idx = np.random.choice(len(parents), p=probs) if probs is not None else np.random.randint(0, len(parents))
                pw, pb, _psc = parents[idx]
            else:
                pw, pb = best_w, best_b
            # Sample target morality value around the DIRECT parent's value, not the average
            if parents:
                y_mu = float(1.0 / (1.0 + np.exp(-(pw + pb))))
            else:
                y_mu = mu_prev
            y_tgt = float(np.random.normal(y_mu, 0.1))
            y_tgt = float(np.clip(y_tgt, 1e-3, 1.0 - 1e-3))
            z = float(np.log(y_tgt / (1.0 - y_tgt)))  # logit
            # Keep parent's weight, adjust bias so w+b=z (makes output match y_tgt when x=1)
            cand_w = pw
            cand_b = z - cand_w
            moral.set_params(cand_w, cand_b)
            alive, score = evaluate()
            # Record morality neuron output for hornet_exists=True
            y = 1.0 / (1.0 + np.exp(-(cand_w * 1.0 + cand_b)))
            cand_vals.append(float(y))
            candidate_series[trial_idx].append(float(y))
            candidate_alive[trial_idx].append(bool(alive))
            if alive:
                cand_scores_alive.append(int(score))
                next_parents.append((cand_w, cand_b, int(score)))
            better = (
                (alive and not best_alive)
                or (alive and score > best_score)
                or ((not best_alive) and (score > best_score))
            )
            if better:
                best_w, best_b = cand_w, cand_b
                best_alive, best_score = alive, score
                improved = True
        moral.set_params(best_w, best_b)
        # Only alive hives produce children in next generation
        parents = next_parents if next_parents else parents or [(best_w, best_b, best_score)]
        # Update prev_mu for the next generation based on this generation's survivors
        if parents:
            prev_mu = float(np.mean([1.0 / (1.0 + np.exp(-(pw + pb))) for pw, pb, _sc in parents]))
        # Average morality value across candidates this generation
        if cand_vals:
            avg_val = float(np.mean(cand_vals))
            moral_values_per_generation.append(avg_val)
            gen_min = min(cand_scores_alive) if cand_scores_alive else None
            survivors = len(cand_scores_alive)
            if gen_min is not None:
                print(
                    f"[moral-1] gen={gen+1} survivors={survivors}/{trials_per_gen} "
                    f"best_alive={best_alive} best_score={best_score} gen_min_alive={gen_min} "
                    f"avg_moral(hornet_exists=True)={avg_val:.4f}"
                )
            else:
                print(
                    f"[moral-1] gen={gen+1} survivors={survivors}/{trials_per_gen} "
                    f"best_alive={best_alive} best_score={best_score} gen_min_alive=None "
                    f"avg_moral(hornet_exists=True)={avg_val:.4f}"
                )
        else:
            print(f"[moral-1] gen={gen+1} best_alive={best_alive} best_score={best_score}")
        # Termination: require ALL hives (candidates) survive this generation
        # AND the minimum score among them meets/exceeds target_score
        if (len(cand_scores_alive) == trials_per_gen) and (min(cand_scores_alive) >= target_score):
            break

    # Plot morality neuron values over generations
    try:
        import matplotlib.pyplot as plt  # type: ignore
        out_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        import datetime as _dt
        fdt = _dt.datetime.now().strftime('%Y%m%d%H%M%S')
        fig_path = os.path.join(out_dir, f'moral_single_values_{fdt}.png')
        plt.figure(figsize=(8, 4))
        # Plot each trial's series as a semi-transparent line
        max_len = max((len(s) for s in candidate_series), default=0)
        x = list(range(max_len))
        for idx, series in enumerate(candidate_series):
            if series:
                alive_flags = candidate_alive[idx]
                # Plot all values (alive or dead) as faint points so full distribution is visible
                plt.scatter(range(len(series)), series, alpha=0.25, s=10, color='gray', zorder=1)
                # Segment by contiguous alive runs; plot runs of length 1 as points, longer as lines
                start = None
                for i, alive in enumerate(alive_flags + [False]):
                    if alive and start is None:
                        start = i
                    elif (not alive) and (start is not None):
                        end = i  # exclusive
                        run_len = end - start
                        xs = list(range(start, end))
                        ys = [series[j] for j in xs]
                        if run_len == 1:
                            plt.scatter(xs, ys, alpha=0.9, s=24, color='gray', zorder=3)
                        else:
                            plt.plot(xs, ys, alpha=0.5, color='gray')
                        start = None
        # Overlay average across generation
        plt.plot(range(len(moral_values_per_generation)), moral_values_per_generation, color='black', linewidth=2, label='avg')
        plt.title('Single-neuron moral values per generation (hornet_exists=True)')
        plt.xlabel('Generation')
        plt.ylabel('Average neuron value')
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"[moral-1] Saved morality values plot to {fig_path}")
    except Exception as e:
        print(f"[moral-1] Could not save morality values plot: {e}")

    return moral

# For CLI compatibility, export Network as the single-neuron moral net
Network = SingleNeuronMoralNet

