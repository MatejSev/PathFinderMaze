#!/usr/bin/env python3
"""
scripts/train_stealth.py — natrénuje nenápadného (stealth) agenta.

Vydělává, ale drží malý spread (1..8) a mírnou korelaci sázky s countem,
aby ho kasino neodhalilo. Míru nenápadnosti ladí --stealth (0=ziskový,
1=nenápadný). Model se uloží podle úrovně (stealth_50.pt apod.).

Příklady:
    python scripts/train_stealth.py                 # vyvážený (0.5)
    python scripts/train_stealth.py --stealth 0.2   # spíš ziskový
    python scripts/train_stealth.py --stealth 0.8   # spíš nenápadný
"""

import _bootstrap  # noqa: F401
import argparse
import torch
from blackjack.players.stealth import StealthPlayer, MAX_STEALTH_BET
from blackjack.core.evaluate import evaluate_player
from blackjack.config import stealth_model


def main():
    ap = argparse.ArgumentParser(description="Trénink stealth agenta")
    ap.add_argument('--stealth', type=float, default=0.5,
                    help='0=ziskový .. 1=nenápadný')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--pool', type=int, default=80000)
    ap.add_argument('--h17', action='store_true')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = stealth_model(args.stealth)
    print(f"Trénink Stealth | stealth={args.stealth} | spread 1-{MAX_STEALTH_BET}"
          f" | {args.steps} kroků | {'H17' if args.h17 else 'S17'} | {dev}")
    player = StealthPlayer(h17=args.h17, stealth=args.stealth,
                           auto_train=False, device=dev)
    player.train(steps=args.steps, pool_size=args.pool)
    torch.save(player.net.state_dict(), out)

    r = evaluate_player(player, n_states=80000, h17=args.h17)
    print(f"\nVýsledek: EV/jednotku={r['ev_unit']*100:+.3f}%  "
          f"EV/kolo={r['ev_round']:+.4f}j")
    print(f"          korelace s countem={r['bet_count_corr']:.2f}  "
          f"spread={r['spread']:.0f}  (nižší = hůř odhalitelný)")
    print(f"\nUloženo do {out}")


if __name__ == "__main__":
    main()
