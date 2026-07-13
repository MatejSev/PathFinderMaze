"""
players/qlearning.py — tabulkový Q-learning agent (sázka podle countu).

Stav = diskretizovaný true count (bucket). Akce = velikost sázky (1..MAX).
Odměna = přesné EV kola * sázka (bezšumové, z core.exact_ev), takže
Q-learning konverguje rychle a spolehlivě k optimální sázecí politice pro
daný count. Hraje basic strategy jako ostatní.

Tabulkový přístup nezvládne spojitý count, proto se true count bucketuje.
Je to hrubší než PPO/DQN, ale plně interpretovatelné a rychle konverguje.

Trénink je součástí třídy (metoda train) — Q-learning je tak lehký, že se
natrénuje za pár sekund přímo při vytvoření hráče (nebo se načte z .npy).
"""

import os
import numpy as np
from blackjack.players.base import Player, MIN_BET, MAX_BET
from blackjack.core.engine import basic_strategy
from blackjack.core.exact_ev import RoundEV
from blackjack.core.environment import (BlackjackEnv, PHASE_BET, PHASE_INSURANCE)

# diskretizace true countu do bucketů
TC_MIN, TC_MAX = -6, 12          # county mimo rozsah se ořežou
N_BUCKETS = TC_MAX - TC_MIN + 1  # jeden bucket na celý bod TC
FULL_13 = np.array([4.] * 13) * 6
HILO = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1], float)


def _tc_bucket(tc):
    b = int(round(tc)) - TC_MIN
    return max(0, min(N_BUCKETS - 1, b))


def _to10(c13):
    c10 = np.zeros(10)
    c10[:8] = c13[:8]
    c10[8] = c13[8:12].sum()
    c10[9] = c13[12]
    return c10


class QLearningPlayer(Player):
    name = "Q-learning"

    def __init__(self, model_path=None, h17=False,
                 auto_train=True, train_states=400000):
        from blackjack.config import QLEARNING_MODEL
        if model_path is None:
            model_path = QLEARNING_MODEL
        self.h17 = h17
        self.n_actions = MAX_BET      # sázky 1..MAX
        self.Q = np.zeros((N_BUCKETS, self.n_actions))
        if os.path.exists(model_path):
            self.Q = np.load(model_path)
        elif auto_train:
            self.train(train_states)
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            np.save(model_path, self.Q)

    def train(self, n_states=400000, alpha=0.05, eps0=0.3, seed=0):
        """Q-learning na bezšumovém EV. Stav je bezkontextový (count bucket),
        takže jde o kontextový bandit — konverguje velmi rychle."""
        env = BlackjackEnv(h17=self.h17, seed=seed)
        rev = RoundEV(h17=self.h17)
        ev_cache = {}
        rng = np.random.default_rng(seed)

        for i in range(n_states):
            env.reset()
            c13 = env.shoe.counts.astype(float).copy()
            cards_left = float(c13.sum())
            dealt = FULL_13 - c13
            tc = float(np.dot(HILO, dealt)) / env.shoe.decks_left()
            b = _tc_bucket(tc)

            rc = int(round(np.dot(HILO, dealt)))
            key = (rc, int(cards_left / 26.0))
            ev = ev_cache.get(key)
            if ev is None:
                ev = rev.compute(_to10(c13))
                ev_cache[key] = ev

            eps = eps0 * (1 - i / n_states)          # klesající explorace
            if rng.random() < eps:
                a = rng.integers(self.n_actions)
            else:
                a = int(np.argmax(self.Q[b]))
            bet = a + 1
            # Kelly/mean-variance odměna: ev*bet - risk*bet^2. Rozptyl roste
            # s bet^2, takže optimální sázka je úměrná výhodě (ne min/max) —
            # jako reálné variabilní sázení.
            risk = 0.0009
            reward = bet * ev - risk * (bet ** 2)
            # bandit update (žádný next-state, kola jsou nezávislá)
            self.Q[b, a] += alpha * (reward - self.Q[b, a])

            # posuň balíček realisticky
            done = False
            while not done:
                k = env.kind()
                if k == PHASE_BET:
                    aa = 0
                elif k == PHASE_INSURANCE:
                    aa = 0
                else:
                    h = env.player_hands[env.active]
                    m = env.play_mask()
                    aa = basic_strategy(h['cards'], env.dealer[0], m)
                    if not m[aa]:
                        aa = 1 if m[1] else 0
                _, r, done = env.step(int(aa))

    def bet(self, bet_obs, env):
        dealt = FULL_13 - env.shoe.counts.astype(float)
        tc = float(np.dot(HILO, dealt)) / env.shoe.decks_left()
        b = _tc_bucket(tc)
        return int(np.argmax(self.Q[b])) + 1
