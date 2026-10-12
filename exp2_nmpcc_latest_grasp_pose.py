#!/usr/bin/env python3
"""手持双臂搬运：保留原命令行的薄入口。"""
from experiments.cli import main as run_experiment


def main():
    run_experiment('handheld', __file__)


if __name__ == '__main__':
    main()
