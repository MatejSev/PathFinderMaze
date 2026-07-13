"""
players/stealth.py — nenápadný agent (advantage play bez vyhození z kasina).

Kasino odhaluje počítače karet podle tří signálů:
  1) korelace sázky s countem (velká sázka = dobrý balíček),
  2) velikost spreadu (poměr max/min sázky),
  3) náhlé skoky v sázkách.

Tento agent vydělává, ALE drží tyto signály nízko. Realizováno dvěma
mechanismy:
  A) TVRDÝ STROP SPREADU: sází jen 1..MAX_STEALTH_BET (např. 1..4) místo
     1..20. Malý spread je sám o sobě mnohem méně nápadný.
  B) PENALIZACE KORELACE při tréninku: reward = EV*sázka - lambda * (jak
     moc sázka koreluje s countem). Agent se tak učí vydělávat, aniž by
     sázel mechanicky podle countu — např. občas vsadí víc i při nízkém
     countu a míň při vysokém, aby signál rozmělnil.

Parametr `stealth` (0..1) ladí přísnost: 0 = skoro jako běžný agent,
1 = maximální nenápadnost (a menší zisk). Výchozí 0.5 = vyvážený kompromis.

Metriku "korelace sázky s countem" počítá compare.py — nízká = nenápadný.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blackjack.players.base import Player
from blackjack.core.engine import basic_strategy
from blackjack.core.exact_ev import RoundEV
from blackjack.core.environment import (BlackjackEnv, PHASE_BET, PHASE_INSURANCE,
                              BET_OBS_DIM)

MAX_STEALTH_BET = 10         # spread 1-10: nejmenší cap, který je ZISKOVÝ
                             # a přitom výrazně diskrétnější než 1-20 běžných
                             # počítacích metod (kompromis zisk vs. nenápadnost)
FULL_13 = np.array([4.] * 13) * 6
HILO = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1], float)


def _to10(c13):
    c10 = np.zeros(10)
    c10[:8] = c13[:8]
    c10[8] = c13[8:12].sum()
    c10[9] = c13[12]
    return c10


class QNet(nn.Module):
    def __init__(self, obs_dim=BET_OBS_DIM, n_actions=MAX_STEALTH_BET, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class StealthPlayer(Player):
    name = "Stealth (nenápadný)"

    def __init__(self, model_path=None, h17=False, stealth=0.5,
                 auto_train=True, device='cpu', train_steps=5000):
        self.device = device
        self.h17 = h17
        self.stealth = stealth
        self.net = QNet().to(device)
        if model_path is None:
            from blackjack.config import stealth_model
            model_path = stealth_model(stealth)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=device))
        elif auto_train:
            self.train(train_steps)
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save(self.net.state_dict(), model_path)
        self.net.eval()

    def train(self, steps=5000, batch=1024, lr=1e-3, seed=0, pool_size=80000):
        env = BlackjackEnv(h17=self.h17, seed=seed)
        rev = RoundEV(h17=self.h17)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        ev_cache = {}

        # pool stavů: (obs, ev, true_count)
        pool_obs = np.zeros((pool_size, BET_OBS_DIM), np.float32)
        pool_ev = np.zeros(pool_size, np.float32)
        pool_tc = np.zeros(pool_size, np.float32)
        for j in range(pool_size):
            env.reset()
            c13 = env.shoe.counts.astype(float).copy()
            pool_obs[j] = env.bet_observation().astype(np.float32)
            dealt = FULL_13 - c13
            pool_tc[j] = float(np.dot(HILO, dealt)) / env.shoe.decks_left()
            rc = int(round(np.dot(HILO, dealt)))
            key = (rc, int(c13.sum() / 26.0))
            ev = ev_cache.get(key)
            if ev is None:
                ev = rev.compute(_to10(c13))
                ev_cache[key] = ev
            pool_ev[j] = ev
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

        bets = torch.arange(1, MAX_STEALTH_BET + 1, device=self.device).float()
        ot_all = torch.as_tensor(pool_obs, device=self.device)
        ev_all = torch.as_tensor(pool_ev, device=self.device)
        tc_all = torch.as_tensor(pool_tc, device=self.device)
        tc_mean = tc_all.mean()
        tc_std = tc_all.std() + 1e-6
        # Nenápadnost zajišťuje hlavně TVRDÝ STROP SPREADU (1..MAX_STEALTH_BET).
        # Penalizace korelace je jen velmi jemná — silnější by agenta stáhla
        # pod nulu. Cíl: být ziskový, ale s menším spreadem než plný counter.
        lam = 0.08 * self.stealth

        for s in range(steps):
            idx = torch.as_tensor(rng.integers(0, pool_size, size=batch),
                                  device=self.device)
            ot = ot_all[idx]; evt = ev_all[idx]; tct = tc_all[idx]
            q = self.net(ot)
            profit = evt[:, None] * bets[None, :]
            tc_norm = ((tct - tc_mean) / tc_std)[:, None]
            bet_dev = (bets[None, :] - bets.mean())
            # jen JEMNÁ penalizace korelace (rozmělní vzorec sázení), bez
            # risk členu — aby agent v rámci capu 1-10 dokázal vsadit dost
            # na dobré county a zůstal ZISKOVÝ.
            penalty = lam * 0.01 * tc_norm * bet_dev
            target = profit - penalty
            loss = F.smooth_l1_loss(q, target)
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 1500 == 0 or s == steps - 1:
                print(f"  [Stealth λ={lam:.2f}] {s}/{steps}  loss={loss.item():.5f}")

    def bet(self, bet_obs, env):
        with torch.no_grad():
            ot = torch.as_tensor(bet_obs.astype(np.float32)[None],
                                 device=self.device)
            q = self.net(ot)[0]
        return int(torch.argmax(q).item()) + 1
