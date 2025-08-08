import argparse
import os

from .moral_evolution import (
    Config,
    Network,
    demo as demo_fn,
    run as evolve_fn,
    write_se_config,
)


def _chdir_to_package_root():
    package_dir = os.path.dirname(__file__)
    os.chdir(package_dir)


def cmd_demo(ip: str | None = None, headless: bool = False):
    _chdir_to_package_root()
    write_se_config()
    cfg_path = os.path.join('config_files', 'morality_layer_config.json')
    c = Config(cfg_path)
    # IP is currently unused by the game logic; accept for future remote modes
    demo_fn(Network(c), viz=not headless)


def cmd_evolve(ip: str | None = None, headless: bool = True):
    _chdir_to_package_root()
    write_se_config()
    cfg_path = os.path.join('config_files', 'morality_layer_config.json')
    c = Config(cfg_path)
    # Run headless by default on servers
    evolve_fn(c, viz=not headless)


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

    args = parser.parse_args()

    if args.command == 'demo':
        cmd_demo(getattr(args, 'ip', None), getattr(args, 'headless', False))
    elif args.command == 'evolve':
        cmd_evolve(getattr(args, 'ip', None), getattr(args, 'headless', True))
    else:
        parser.error('Unknown command')


if __name__ == '__main__':
    main()
