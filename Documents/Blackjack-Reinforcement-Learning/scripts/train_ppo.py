#!/usr/bin/env python3
"""
scripts/train_ppo.py — natrénuje hlavního RL agenta (PPO).

Sázka + insurance + hit/stand/double/split. Warm-start + PPO doladění;
nejlepší model se vybírá podle PŘESNÉHO (bezšumového) EV, ne podle šumu.

Příklady:
    python scripts/train_ppo.py                 # výchozí (800 iterací)
    python scripts/train_ppo.py --quick         # rychlý zkušební běh
    python scripts/train_ppo.py --iters 1500    # delší trénink
    python scripts/train_ppo.py --h17
"""

import _bootstrap  # noqa: F401
import argparse
from blackjack.training.ppo import train_ppo


def main():
    ap = argparse.ArgumentParser(description="Trénink PPO agenta")
    ap.add_argument('--quick', action='store_true', help='rychlý zkušební běh')
    ap.add_argument('--iters', type=int, default=None, help='počet PPO iterací')
    ap.add_argument('--envs', type=int, default=1024, help='paralelní prostředí')
    ap.add_argument('--h17', action='store_true', help='krupiér táhne na soft 17')
    args = ap.parse_args()

    if args.quick:
        train_ppo(iters=args.iters or 120, n_envs=512, warm=500,
                  eval_every=30, eval_states=60000, h17=args.h17)
    else:
        train_ppo(iters=args.iters or 800, n_envs=args.envs, h17=args.h17)


if __name__ == "__main__":
    main()
