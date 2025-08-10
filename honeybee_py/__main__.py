import argparse
import os

from .moral_evolution import (
    Config,
    Network,
    demo as demo_fn,
    run as evolve_fn,
)


def _chdir_to_package_root():
    package_dir = os.path.dirname(__file__)
    os.chdir(package_dir)


def cmd_demo(ip: str | None = None, headless: bool = False):
    _chdir_to_package_root()
    c = Config(config_file='')
    # IP is currently unused by the game logic; accept for future remote modes
    try:
        demo_fn(Network(c), viz=not headless)
    except (ImportError, ModuleNotFoundError) as e:
        print("[demo] TensorFlow/Keras not available. Install tensorflow to run the demo.")
        raise
    except FileNotFoundError as e:
        print("[demo] Missing model files. Train models via 'python -m honeybee_py train-regular' and 'python -m honeybee_py train-hornet'.")
        raise


def cmd_evolve(ip: str | None = None, headless: bool = True):
    _chdir_to_package_root()
    c = Config(config_file='')
    # Run headless by default on servers
    try:
        evolve_fn(c, viz=not headless)
    except (ImportError, ModuleNotFoundError):
        print("[evolve] TensorFlow/Keras not available. Install tensorflow to run evolution.")
        raise
    except FileNotFoundError:
        print("[evolve] Missing model files. Train models via 'python -m honeybee_py train-regular' and 'python -m honeybee_py train-hornet'.")
        raise


def main():
    parser = argparse.ArgumentParser(
        prog='python -m honeybee_py',
        description='Honeybee simulation commands',
    )
    sub = parser.add_subparsers(dest='command', required=True)
    p_demo = sub.add_parser('demo', help='Run a quick visual demo')
    p_demo.add_argument('ip', nargs='?', help='Server IP (for future remote modes)')
    p_demo.add_argument('--headless', action='store_true', help='Disable visualization (useful on servers)')

    p_evo = sub.add_parser('evolve', help='Run moral evolution loop')
    p_evo.add_argument('ip', nargs='?', help='Server IP (for future remote modes)')
    p_evo.add_argument('--headless', action='store_true', help='Disable visualization (default on servers)')

    p_tr = sub.add_parser('train-regular', help='Train the regular policy model and save to keras_models/')
    p_tr.add_argument('--epochs', type=int, default=32)
    p_tr.add_argument('--dataset-size', type=int, default=200_000)
    p_tr.add_argument('--batch-size', type=int, default=1024)
    p_tr.add_argument('--board-size', type=int, default=20)
    p_tr.add_argument('--mixed-precision', action='store_true')

    p_th = sub.add_parser('train-hornet', help='Train the hornet confrontation model and save to keras_models/')
    p_th.add_argument('--epochs', type=int, default=100)
    p_th.add_argument('--dataset-size', type=int, default=200_000)
    p_th.add_argument('--batch-size', type=int, default=1024)
    p_th.add_argument('--board-size', type=int, default=20)

    args = parser.parse_args()

    if args.command == 'demo':
        cmd_demo(getattr(args, 'ip', None), getattr(args, 'headless', False))
    elif args.command == 'evolve':
        cmd_evolve(getattr(args, 'ip', None), getattr(args, 'headless', True))
    elif args.command == 'train-regular':
        _chdir_to_package_root()
        from .train_regular_no_hornets import train_regular_policy  # lazy import
        train_regular_policy(
            epochs=args.epochs,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            board_size=args.board_size,
            use_mixed_precision=args.mixed_precision,
        )
    elif args.command == 'train-hornet':
        _chdir_to_package_root()
        from .train_hornet_confront import train as train_hornet_policy  # lazy import
        train_hornet_policy(
            epochs=args.epochs,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            board_size=args.board_size,
        )
    else:
        parser.error('Unknown command')


if __name__ == '__main__':
    main()
