#!/usr/bin/env python3
"""
scripts/compare.py — porovná všechny hráče (bezšumově, přesné EV složení).

Všichni hráči mají stejné rozhraní, hrají basic strategy a sázejí plynule
podle síly countu — liší se jen KVALITOU sázení. Vyhodnocení je bez šumu
(přesné EV složení balíčku), takže rozdíly na setiny % jsou spolehlivé.

Sloupce: EV/jednotku (kvalita počítání), korel.count (nápadnost — jak moc
sázka kopíruje count) a spread (poměr max/min sázky).

Příklady:
    python scripts/compare.py                          # počítací metody + Perfect
    python scripts/compare.py --ppo                    # + PPO agent
    python scripts/compare.py --all                    # + Q-learning, DQN, Stealth
    python scripts/compare.py --all --ppo --states 200000
"""

import _bootstrap  # noqa: F401
import argparse
import os
from blackjack.core.evaluate import evaluate_player
from blackjack.players import (HiLoPlayer, KOPlayer, HiOpt2Player,
                               PerfectPlayer)
from blackjack.config import PPO_MODEL


def main():
    ap = argparse.ArgumentParser(description="Porovnání hráčů")
    ap.add_argument('--states', type=int, default=150000)
    ap.add_argument('--h17', action='store_true')
    ap.add_argument('--ppo', action='store_true', help='přidat PPO agenta')
    ap.add_argument('--all', action='store_true',
                    help='přidat Q-learning, DQN a Stealth')
    args = ap.parse_args()

    players = [HiLoPlayer(), KOPlayer(), HiOpt2Player(),
               PerfectPlayer(h17=args.h17)]

    if args.ppo:
        if os.path.exists(PPO_MODEL):
            from blackjack.players import RLAgentPlayer
            players.append(RLAgentPlayer(PPO_MODEL, device='cpu'))
        else:
            print(f"PPO model nenalezen ({PPO_MODEL}) — přeskočen. "
                  f"Natrénuj: python scripts/train_ppo.py\n")

    if args.all:
        from blackjack.players import (QLearningPlayer, DQNPlayer,
                                       StealthPlayer)
        print("Připravuji Q-learning / DQN / Stealth "
              "(trénink při prvním spuštění)…")
        players.append(QLearningPlayer(h17=args.h17))
        players.append(DQNPlayer(h17=args.h17))
        players.append(StealthPlayer(h17=args.h17, stealth=0.5))

    print("=" * 88)
    print("  POROVNÁNÍ HRÁČŮ — bezšumové (přesné EV složení balíčku)")
    print(f"  6 balíčků | {'H17' if args.h17 else 'S17'} | "
          f"{args.states:,} stavů | plynulé sázení dle countu")
    print("=" * 88)
    print(f"\n  {'hráč':<20}{'EV/kolo':>11}{'EV/jednotku':>13}{'±CI95':>9}"
          f"{'prům.sázka':>11}{'korel.count':>12}{'spread':>8}")
    print("  " + "-" * 84)

    rows = []
    for p in players:
        r = evaluate_player(p, n_states=args.states, h17=args.h17)
        rows.append((p.name, r))
        print(f"  {p.name:<20}{r['ev_round']:>+10.4f}j{r['ev_unit']*100:>+12.3f}%"
              f"{r['ci']*100:>8.3f}%{r['avg_bet']:>10.2f}j"
              f"{r['bet_count_corr']:>12.2f}{r['spread']:>8.0f}")
    print("=" * 88)

    perfect = next((r for n, r in rows if n.startswith('Perfect')), None)
    print("\n  Jak číst:")
    print("  • EV/jednotku = kvalita počítání (normalizováno na sázku).")
    print("  • korel.count = jak moc sázka kopíruje count (vysoká = nápadná).")
    print("  • spread = poměr max/min sázky (velký = nápadný).")
    if perfect:
        over = [n for n, r in rows
                if r['ev_unit'] > perfect['ev_unit'] + max(r['ci'], perfect['ci'])]
        if over:
            print(f"  • POZOR: {', '.join(over)} přesahuje Perfect — chyba měření.")
        else:
            print("  • Nikdo nepřekonává Perfect nad rámec šumu — správně.")
    print("  • Perfect je STROP. RL/Q/DQN počítání dorovnají, nepřekročí.")
    print("  • Stealth: nižší EV, ale i nižší korelace a spread (nenápadný).")


if __name__ == "__main__":
    main()
