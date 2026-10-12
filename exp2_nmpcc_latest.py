#!/usr/bin/env python3
"""平放双臂搬运：保留原命令行的薄入口。"""
from experiments.cli import main as run_experiment


def main():
    run_experiment('flat', __file__)


if __name__ == '__main__':
    main()
