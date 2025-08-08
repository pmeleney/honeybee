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


def cmd_demo():
    _chdir_to_package_root()
    write_se_config()
    cfg_path = os.path.join('config_files', 'morality_layer_config.json')
    c = Config(cfg_path)
    demo_fn(Network(c))


def cmd_evolve():
    _chdir_to_package_root()
    write_se_config()
    cfg_path = os.path.join('config_files', 'morality_layer_config.json')
    c = Config(cfg_path)
    evolve_fn(c)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m honeybee_py',
        description='Honeybee simulation commands',
    )
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('demo', help='Run a quick visual demo')
    sub.add_parser('evolve', help='Run moral evolution loop')

    args = parser.parse_args()

    if args.command == 'demo':
        cmd_demo()
    elif args.command == 'evolve':
        cmd_evolve()
    else:
        parser.error('Unknown command')


if __name__ == '__main__':
    main()
