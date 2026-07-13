"""
players/perfect.py — teoreticky optimální hráč (strop).

Sází podle PŘESNÉ okamžité výhody spočtené z aktuálního složení balíčku
(core.exact_ev). To je maximum informace, co v kartách je — žádný count
systém to nemůže překonat. Aby se srovnával jen kvalitou informace (ne
agresivitou sázení), převede přesnou výhodu na ekvivalentní true count a
použije STEJNÝ spread jako počítací metody.
"""

import numpy as np
from blackjack.players.base import Player, bet_from_true_count
from blackjack.core.exact_ev import RoundEV

FULL_13 = np.array([4.] * 13) * 6

# převod 13 ranků -> 10 hodnotových kategorií pro exact EV
def _to10(c13):
    c10 = np.zeros(10)
    c10[:8] = c13[:8]
    c10[8] = c13[8:12].sum()
    c10[9] = c13[12]
    return c10

# EV ≈ 0.0048*(TC - 2)  => ekvivalentní TC = EV/0.0048 + 2
EV_PER_TC = 0.0048


class PerfectPlayer(Player):
    name = "Perfect (EoR)"

    def __init__(self, h17=False):
        self._rev = RoundEV(h17=h17)
        self._cache = {}

    def _edge(self, env):
        c13 = env.shoe.counts.astype(float)
        cards_left = float(c13.sum())
        dealt = FULL_13 - c13
        # cache na (Hi-Lo running count, hloubka) — EV je jejich funkcí
        rc = int(round(np.dot(
            np.array([1,1,1,1,1,0,0,0,-1,-1,-1,-1,-1], float), dealt)))
        dl = int(cards_left / 26.0)
        key = (rc, dl)
        ev = self._cache.get(key)
        if ev is None:
            ev = self._rev.compute(_to10(c13))
            self._cache[key] = ev
        return ev

    def bet(self, bet_obs, env):
        # Perfect zná PŘESNOU výhodu složení. Převede ji na ekvivalentní
        # sílu countu (EV ≈ 0.0048*(TC-2) => TC ≈ EV/0.0048 + 2) a použije
        # STEJNÝ plynulý ramp jako počítací metody. Protože má nejpřesnější
        # odhad výhody, je zaručeným stropem.
        from blackjack.players.base import bet_from_edge_strength
        ev = self._edge(env)
        strength = ev / EV_PER_TC + 2.0
        return bet_from_edge_strength(strength)
