## Honeybee Moral Decision-Making Simulation

A multi-agent honeybee simulation exploring moral decision-making between foraging and defense. Bees use specialized neural networks for foraging and hornet confrontation, and a simple “moral” controller decides which behavior to prioritize each turn. Includes pretrained Keras models, a custom evolution loop for the moral layer, and optional NEAT tooling.

Preview: see `DEMO_honeybeegame.mov` in the repo root.

### Key Features
- **Interactive simulation**: Queen, bees, flowers, and hornets on a grid with live visualization
- **Neural control**:
  - Regular net (foraging; 7 inputs → 4-way move)
  - Hornet net (confrontation; 4 inputs → 4-way move)
  - Moral net (1-bit context → choose regular vs hornet outputs)
- **Pretrained models included**: Ready-to-run `.keras` models plus CSV weights
- **Evolution loop**: Custom evolution of the moral layer via repeated gameplay
- **Modular codebase**: Game engine separated from training and configs

### Repository Layout
```
honeybee/
├── honeybee_py/
│   ├── game/
│   │   ├── game.py               # Game loop, viz, I/O with networks
│   │   ├── gameobjects.py        # Bee, Queen, Flower, Hornet
│   │   ├── helpers.py            # Utilities
│   │   └── honeybeeconfig.py     # Static game settings
│   ├── config_files/             # JSON configs for networks
│   ├── keras_models/             # Pretrained Keras models (regular/hornet)
│   ├── best_weights_and_biases/  # Layer CSVs for both nets
│   ├── moral_evolution.py        # Evolutionary loop for the moral layer
│   ├── train_bee_hornet_networks.py # (Re)train and export Keras models/CSVs
│   ├── neat/
│   │   └── neat_evolve.py        # Optional NEAT tooling (experimental)
│   └── demonstration.py          # Legacy NEAT demo (see notes)
├── DEMO_honeybeegame.mov
├── LICENSE
└── README.md
```

## Installation

### Prerequisites
- Python 3.9–3.11 recommended
- pip

### Install
```bash
git clone https://github.com/pmeleney/honeybee.git
cd honeybee
pip install -r requirements.txt
```

Notes for Apple Silicon (M-series):
- If standard `tensorflow` fails to install, prefer:
  - `pip install tensorflow-macos tensorflow-metal`
- Keep the rest of the requirements the same.

## Quickstart

### Local (visual)
```bash
python -m honeybee_py demo
```

### Server/EC2 (headless, optional IP arg)
```bash
python -m honeybee_py demo 203.0.113.10 --headless
```

What you’ll see:
- Bees pursue flowers and return food to the queen
- Hornets spawn periodically and move toward the queen
- The moral net toggles between regular vs hornet-control based on hornet presence

## Training and Evolution

### 1) Re-train or regenerate the control nets (optional)
Exports both `.keras` models and `best_weights_and_biases` CSVs.
```bash
cd honeybee_py
python train_bee_hornet_networks.py
```

### 2) Evolve the moral layer
Runs repeated games, prunes poor performers, mutates survivors, and continues until the desired count of “alive” networks is reached.
```bash
python -m honeybee_py evolve --headless
```
Outputs are printed to console; pretrained regular/hornet models are loaded from `keras_models/`.

### 3) (Experimental) NEAT tooling
The `neat/` folder contains an example driver. You must provide a NEAT config file and ensure output directories exist.
```bash
cd honeybee_py/neat
mkdir -p outputs/figures outputs/winners checkpoints
python neat_evolve.py <path-to-neat-config>
```

Legacy note: `honeybee_py/demonstration.py` was written for an older `play_game` signature (NEAT-only). It’s kept for reference but not wired to the current game API.

## How It Works

### Game objects
- **Queen**: 2x2 area at board center; bees deliver food here
- **Bee**: Moves one tile per turn; collects food, confronts hornets when appropriate
- **Flower**: Spawns with food; color darkens when depleted
- **Hornet**: Spawns at edges and moves toward the queen; on overlap, queen dies

### Neural networks
- **Regular net (Keras)**: Inputs = [bee(x,y), flower(x,y), queen(x,y), has_food] scaled to grid; outputs 4-way move probabilities
- **Hornet net (Keras)**: Inputs = [bee(x,y), hornet(x,y)]; outputs 4-way move probabilities
- **Moral net (custom)**: Input = boolean hornet_exists; simple linear layer decides whether to use regular vs hornet outputs

### Control loop
Per turn, for each bee:
1. Build inputs for both control nets
2. Run Keras models to get 4-way moves
3. Run moral net; if output > 0.5 choose hornet move, else regular move
4. Apply move with edge/corner corrections; update scores and overlaps

## Configuration

Edit `honeybee_py/game/honeybeeconfig.py` to change board size, counts, and pacing (e.g., hornet frequency).

Network/config JSONs in `honeybee_py/config_files/`:
- `morality_layer_config.json`: moral layer structure and evolution params (e.g., `num_nets`, init/update distributions)
- `regular_network_config.json`: shape/activations + CSV paths for regular net layers
- `hornet_network_config.json`: shape/activations + CSV paths for hornet net layers

`moral_evolution.write_se_config()` will (re)write defaults into this directory if needed.

## Troubleshooting
- Matplotlib window doesn’t show: ensure you run locally with an interactive backend; or remove `viz=True` to run headless.
- TensorFlow install issues on macOS: use `tensorflow-macos` + `tensorflow-metal` as noted above.
- NEAT run errors when saving: create `outputs/figures` and `outputs/winners` directories first.

## License
MIT — see `LICENSE`.

## Citation
If you use this project in academic work:
```bibtex
@software{honeybee_simulation,
  title={Honeybee Simulation: Moral Decision-Making in Multi-Agent Systems},
  author={Meleney, Peter},
  year={2024},
  url={https://github.com/pmeleney/honeybee}
}
```

## Contact
- Author: pmeleney
- Repo: `https://github.com/pmeleney/honeybee`
- Issues: `https://github.com/pmeleney/honeybee/issues`