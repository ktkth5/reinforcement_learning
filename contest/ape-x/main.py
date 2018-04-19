import argparse
import retro
from retro_contest import make

import torch
import torch.nn as nn

from utils import get_screen

parser = argparse.ArgumentParser(description='Run Ape-X on Sonic')
parser.add_argument("--seed", default=13, type=int)
parser.add_argument("--epochs", default=10, type=int)


def main():
    global args
    args = parser.parse_args()

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    current_state = get_screen(env)

    for epoch in range(args.epochs):
        env.reset()
