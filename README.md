# Honeybee Simulation

A sophisticated multi-agent simulation exploring moral decision-making in honeybee colonies using neural networks and evolutionary algorithms.

![Honeybee Game Demo](DEMO_honeybeegame.mov)

## Overview

This project simulates a honeybee colony where bees must make moral decisions about whether to prioritize food collection or hornet avoidance. The simulation uses multiple neural networks:

- **Regular Network**: Handles normal foraging behavior
- **Hornet Network**: Manages hornet avoidance strategies  
- **Moral Network**: Makes high-level decisions about which behavior to prioritize

The system employs evolutionary algorithms (NEAT) and reinforcement learning to develop intelligent, adaptive behaviors in the simulated bees.

## Features

- **Multi-Agent Simulation**: Realistic honeybee colony behavior with queens, workers, flowers, and hornets
- **Neural Network Evolution**: Uses NEAT algorithm to evolve intelligent bee behaviors
- **Moral Decision Making**: Explores ethical decision-making in artificial agents
- **Visualization**: Real-time game visualization with matplotlib
- **Modular Architecture**: Clean separation between game logic, neural networks, and training systems

## Project Structure

```
honeybee/
├── honeybee_py/                    # Main Python package
│   ├── game/                       # Game engine and objects
│   │   ├── game.py                # Main game logic and simulation
│   │   ├── gameobjects.py         # Bee, Queen, Flower, Hornet classes
│   │   ├── helpers.py             # Utility functions
│   │   └── honeybeeconfig.py      # Game configuration
│   ├── neat/                      # NEAT evolution
│   │   └── neat_evolve.py         # NEAT algorithm implementation
│   ├── train_bee_hornet_networks/ # Neural network training
│   ├── best_weights_and_biases/   # Pre-trained network weights
│   ├── keras_models/              # Saved Keras models
│   ├── config_files/              # Configuration files
│   ├── training_checkpoints/      # Training checkpoints
│   ├── moral_evolution.py         # Moral network evolution
│   ├── train_bee_hornet_networks.py # Network training scripts
│   └── demonstration.py           # Demo script
├── LICENSE                         # MIT License
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Dependencies

Install the required packages:

```bash
pip install numpy matplotlib tensorflow scipy scikit-learn neat-python
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pmeleney/honeybee.git
cd honeybee
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Demo

To see the simulation in action:

```bash
cd honeybee_py
python demonstration.py <winner_path> <config_path>
```

### Training Networks

1. **Train Regular and Hornet Networks**:
```bash
python train_bee_hornet_networks.py
```

2. **Run Moral Evolution**:
```bash
python moral_evolution.py
```

3. **NEAT Evolution**:
```bash
python neat/neat_evolve.py <config_file>
```

### Game Components

#### Game Objects

- **Queen**: Central figure that bees return food to (2x2 area in center)
- **Bees**: Workers that collect food and make moral decisions
- **Flowers**: Food sources that bees collect from
- **Hornets**: Threats that attack bees and the queen

#### Neural Networks

- **Regular Network**: 7 inputs (bee position, flower position, queen position, has_food)
- **Hornet Network**: 4 inputs (bee position, hornet position)
- **Moral Network**: 1 input (hornet_exists) → 1 output (decision threshold)

#### Game Mechanics

1. Bees start in random positions
2. Each turn, bees decide whether to:
   - Collect food from nearest flower (regular behavior)
   - Avoid hornets (defensive behavior)
3. Moral network determines which behavior to prioritize
4. Bees score points for successful food collection
5. Game ends when queen dies or max turns reached

## Configuration

### Game Settings

Edit `honeybee_py/game/honeybeeconfig.py` to modify:

- Board size and layout
- Number of bees, flowers, hornets
- Game duration and scoring
- Hornet behavior patterns

### Network Training

Configuration files in `honeybee_py/config_files/`:

- `morality_layer_config.json`: Moral network parameters
- `regular_network_config.json`: Regular behavior network
- `hornet_network_config.json`: Hornet avoidance network

## Research Applications

This simulation is designed for research in:

- **Artificial Moral Agents**: Exploring ethical decision-making in AI
- **Multi-Agent Systems**: Emergent behavior in social insects
- **Evolutionary Algorithms**: NEAT for complex behavior evolution
- **Reinforcement Learning**: Reward-based learning in dynamic environments

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NEAT algorithm implementation based on the `neat-python` library
- TensorFlow/Keras for neural network training
- Matplotlib for visualization
- Inspired by research in artificial moral agents and swarm intelligence

## Citation

If you use this code in your research, please cite:

```bibtex
@software{honeybee_simulation,
  title={Honeybee Simulation: Moral Decision-Making in Multi-Agent Systems},
  author={pmeleney},
  year={2024},
  url={https://github.com/pmeleney/honeybee}
}
```

## Contact

- **Author**: pmeleney
- **Repository**: https://github.com/pmeleney/honeybee
- **Issues**: https://github.com/pmeleney/honeybee/issues

---

*This project explores the intersection of artificial intelligence, moral philosophy, and biological inspiration through the lens of honeybee behavior.* 