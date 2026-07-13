#!/usr/bin/env python3
"""
scripts/train_qlearning.py — natrénuje tabulkového Q-learning agenta.

Učí se na přesném (bezšumovém) EV složení balíčku, konverguje rychle.
Sázka roste plynule podle síly countu (Kelly / mean-variance cíl).

Příklady:
    python scripts/train_qlearning.py
    python scripts/train_qlearning.py --states 800000
    python scripts/train_qlearning.py --h17
"""

import _bootstrap  # noqa: F401
import argparse
import numpy as np
from blackjack.players.qlearning import QLearningPlayer, _tc_bucket
from blackjack.config import QLEARNING_MODEL


def main():
    ap = argparse.ArgumentParser(description="Trénink Q-learning agenta")
    ap.add_argument('--states', type=int, default=400000)
    ap.add_argument('--h17', action='store_true')
    args = ap.parse_args()

    print(f"Trénink Q-learning | {args.states:,} stavů | "
          f"{'H17' if args.h17 else 'S17'}")
    player = QLearningPlayer(h17=args.h17, auto_train=False)
    player.train(n_states=args.states)
    np.save(QLEARNING_MODEL, player.Q)

    print("\nNaučená sázecí politika podle true countu:")
    for tc in range(-3, 10):
        bet = int(np.argmax(player.Q[_tc_bucket(tc)])) + 1
        print(f"  TC={tc:+d} -> sázka {bet}")
    print(f"\nUloženo do {QLEARNING_MODEL}")


if __name__ == "__main__":
    main()
