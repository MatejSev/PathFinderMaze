"""
scripts/analyze_stealth.py — kompromis mezi ziskem a nenápadností.

Ukazuje jádro celého advantage play: čím menší bet spread (a tedy méně
nápadný hráč), tím nižší zisk. Malý spread, který kasino nerozezná od
rekreačního hráče, je v 6-balíčkové hře prodělečný; zisk začíná až u
většího spreadu — přesně toho, podle kterého kasino počítače karet odhalí.

Měří se bezšumově (přesné EV složení). Strategie: sázej MAX při true
countu nad prahem, jinak minimum (1). Prochází různé stropy spreadu.

Spuštění:
    python stealth_analysis.py
    python stealth_analysis.py --states 150000
"""

import _bootstrap  # noqa: F401

import argparse
import numpy as np
from blackjack.core.evaluate import evaluate_player
from blackjack.players.base import Player

FULL_13 = np.array([4.] * 13) * 6
HILO = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1], float)


class _SpreadPlayer(Player):
    """Bet MAX při TC > práh, jinak 1. Slouží k proměření tradeoffu."""
    def __init__(self, max_bet, threshold=2.0):
        self.max_bet = max_bet
        self.threshold = threshold
        self.name = f"spread 1-{max_bet}"

    def bet(self, bet_obs, env):
        dealt = FULL_13 - env.shoe.counts.astype(float)
        tc = float(np.dot(HILO, dealt)) / env.shoe.decks_left()
        return self.max_bet if tc > self.threshold else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', type=int, default=120000)
    ap.add_argument('--h17', action='store_true')
    args = ap.parse_args()

    print("=" * 74)
    print("  KOMPROMIS ZISK vs. NENÁPADNOST (bet spread)")
    print(f"  6 balíčků | {'H17' if args.h17 else 'S17'} | "
          f"{args.states:,} stavů | basic strategy")
    print("=" * 74)
    print(f"\n  {'spread':<10}{'EV/jednotku':>14}{'EV/kolo':>12}"
          f"{'korel.count':>13}{'nápadnost':>12}")
    print("  " + "-" * 66)

    for maxb in [2, 3, 4, 6, 8, 10, 12, 16, 20]:
        r = evaluate_player(_SpreadPlayer(maxb), n_states=args.states,
                            h17=args.h17)
        # nápadnost: kombinace spreadu a korelace (hrubý index 0..10)
        detect = min(10, (maxb / 2) * abs(r['bet_count_corr']))
        flag = "nízká" if detect < 3 else ("střední" if detect < 6 else "VYSOKÁ")
        prof = " zisk" if r['ev_unit'] > 0 else "ztráta"
        print(f"  1-{maxb:<8}{r['ev_unit']*100:>+13.3f}%{r['ev_round']:>+11.4f}j"
              f"{r['bet_count_corr']:>13.2f}{flag:>12}   [{prof}]")

    print("=" * 74)
    print("\n  Čtení:")
    print("  • Malý spread (1-2 až 1-4): nenápadný, ale ZTRÁTOVÝ.")
    print("  • Zisk začíná kolem 1-8, solidní až 1-12 a výš.")
    print("  • Jenže velký spread = vysoká nápadnost = kasino tě odhalí.")
    print("  • To je jádro problému: nenápadnost a zisk jsou v konfliktu.")
    print("    Právě proto je počítání karet v praxi tak těžké provozovat.")


if __name__ == "__main__":
    main()
