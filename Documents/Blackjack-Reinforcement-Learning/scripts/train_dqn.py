#!/usr/bin/env python3
"""
scripts/train_dqn.py — natrénuje Deep Q-Network agenta pro sázení.

Bere celé bet-observation (count, hloubka, hustota es i desítek). Sázka
plynulá podle síly countu (Kelly / mean-variance cíl). Trénink na přesném EV.

Příklady:
    python scripts/train_dqn.py
    python scripts/train_dqn.py --steps 12000
    python scripts/train_dqn.py --h17
"""

import _bootstrap  # noqa: F401
import argparse
import numpy as np
import torch
from blackjack.players.dqn import DQNPlayer
from blackjack.config import DQN_MODEL


def main():
    ap = argparse.ArgumentParser(description="Trénink DQN agenta")
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--pool', type=int, default=60000)
    ap.add_argument('--h17', action='store_true')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Trénink DQN | {args.steps} kroků | pool {args.pool:,} | "
          f"{'H17' if args.h17 else 'S17'} | {dev}")
    player = DQNPlayer(h17=args.h17, auto_train=False, device=dev)
    player.train(steps=args.steps, pool_size=args.pool)
    torch.save(player.net.state_dict(), DQN_MODEL)

    print("\nNaučená sázecí politika podle true countu:")
    for tc in range(-3, 10):
        o = np.array([tc/10., np.tanh(tc/5.), 0.5,
                      1. if tc >= 2 else 0., 0., 0.], np.float32)
        with torch.no_grad():
            q = player.net(torch.as_tensor(o[None], device=dev))[0]
        print(f"  TC={tc:+d} -> sázka {int(torch.argmax(q)) + 1}")
    print(f"\nUloženo do {DQN_MODEL}")


if __name__ == "__main__":
    main()
