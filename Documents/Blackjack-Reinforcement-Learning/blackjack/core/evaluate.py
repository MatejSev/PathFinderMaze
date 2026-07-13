"""
core/evaluate.py — bezšumové vyhodnocení kvality SÁZENÍ libovolného hráče.

Místo hraní milionů náhodných kol (obrovský rozptyl) projdeme reálné stavy
balíčku a pro každý spočítáme PŘESNÉ EV kola (exact_ev). Hráčova hodnota =
průměr přesných EV vážený jeho sázkami. Rozptyl je téměř nulový, takže i
setiny procenta jsou spolehlivé — a keep-best v tréninku se řídí pravdou,
ne šťastným vzorkem.

Vrací: EV/jednotku (kvalita sázení), EV/kolo, průměrnou sázku.
"""

import numpy as np
from blackjack.core.environment import (BlackjackEnv, PHASE_BET, PHASE_INSURANCE,
                              N_PLAY_ACTIONS)
from blackjack.core.engine import basic_strategy
from blackjack.core.exact_ev import RoundEV

FULL_13 = np.array([4.] * 13) * 6
HILO = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1], float)


def _to10(c13):
    c10 = np.zeros(10)
    c10[:8] = c13[:8]
    c10[8] = c13[8:12].sum()
    c10[9] = c13[12]
    return c10


def evaluate_player(player, n_states=100000, h17=False, seed=999,
                    n_blocks=100):
    """Bezšumové EV/jednotku hráče. Vrací (ev_unit, ev_round, avg_bet, ci)."""
    env = BlackjackEnv(h17=h17, seed=seed)
    rev = RoundEV(h17=h17)
    ev_cache = {}

    sum_w = 0.0
    sum_b = 0.0
    bs = max(1, n_states // n_blocks)
    blocks = []
    cw = cb = 0.0
    # pro korelaci sázky s true countem (nápadnost)
    bets_arr = []
    tcs_arr = []

    for i in range(n_states):
        env.reset()
        c13 = env.shoe.counts.astype(float).copy()
        cards_left = float(c13.sum())
        dealt = FULL_13 - c13

        rc = int(round(np.dot(HILO, dealt)))
        dl = int(cards_left / 26.0)
        key = (rc, dl)
        ev = ev_cache.get(key)
        if ev is None:
            ev = rev.compute(_to10(c13))
            ev_cache[key] = ev

        bet = player.bet(env.bet_observation(), env)
        tc = float(np.dot(HILO, dealt)) / env.shoe.decks_left()

        sum_w += bet * ev
        sum_b += bet
        cw += bet * ev
        cb += bet
        bets_arr.append(bet)
        tcs_arr.append(tc)
        if (i + 1) % bs == 0:
            blocks.append(cw / max(cb, 1e-9))
            cw = cb = 0.0

        # dohraj kolo basic strategy, ať se balíček realisticky posouvá
        done = False
        while not done:
            k = env.kind()
            if k == PHASE_BET:
                a = 0
            elif k == PHASE_INSURANCE:
                a = 0
            else:
                h = env.player_hands[env.active]
                m = env.play_mask()
                a = basic_strategy(h['cards'], env.dealer[0], m)
                if not m[a]:
                    a = 1 if m[1] else 0
            _, r, done = env.step(int(a))

    ev_unit = sum_w / sum_b if sum_b else 0.0
    ev_round = sum_w / n_states
    avg_bet = sum_b / n_states
    arr = np.array(blocks)
    ci = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    # nápadnost: korelace sázky s true countem + spread
    b = np.array(bets_arr, float)
    t = np.array(tcs_arr, float)
    if b.std() > 1e-9 and t.std() > 1e-9:
        corr = float(np.corrcoef(b, t)[0, 1])
    else:
        corr = 0.0
    spread = float(b.max() / max(b.min(), 1))
    return {
        'ev_unit': ev_unit, 'ev_round': ev_round, 'avg_bet': avg_bet,
        'ci': ci, 'bet_count_corr': corr, 'spread': spread,
    }
