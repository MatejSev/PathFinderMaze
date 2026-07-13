"""
players/dqn.py — Deep Q-Network agent pro sázení.

Na rozdíl od tabulkového Q-learningu bere DQN CELÉ bet-observation (true
count, hloubka, hustota es i desítek), takže může sázet jemněji a využít
víc informace. Q-síť odhaduje hodnotu každé velikosti sázky; hraje se
argmax. Trénink na bezšumovém EV (core.exact_ev) — kontextový bandit, takže
stačí jednoduchý DQN bez replay/target-net triků (kola jsou nezávislá).

Optimalizace: dvě skryté vrstvy, Huber loss, Adam, klesající explorace.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blackjack.players.base import Player, MAX_BET
from blackjack.core.engine import basic_strategy
from blackjack.core.exact_ev import RoundEV
from blackjack.core.environment import (BlackjackEnv, PHASE_BET, PHASE_INSURANCE,
                              BET_OBS_DIM)

FULL_13 = np.array([4.] * 13) * 6
HILO = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1], float)


def _to10(c13):
    c10 = np.zeros(10)
    c10[:8] = c13[:8]
    c10[8] = c13[8:12].sum()
    c10[9] = c13[12]
    return c10


class QNet(nn.Module):
    def __init__(self, obs_dim=BET_OBS_DIM, n_actions=MAX_BET, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNPlayer(Player):
    name = "Deep Q (DQN)"

    def __init__(self, model_path=None, h17=False,
                 auto_train=True, device='cpu', train_steps=6000):
        from blackjack.config import DQN_MODEL
        if model_path is None:
            model_path = DQN_MODEL
        self.device = device
        self.h17 = h17
        self.net = QNet().to(device)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=device))
        elif auto_train:
            self.train(train_steps)
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save(self.net.state_dict(), model_path)
        self.net.eval()

    def train(self, steps=6000, batch=512, lr=1e-3, eps0=0.3, seed=0,
              pool_size=60000):
        env = BlackjackEnv(h17=self.h17, seed=seed)
        rev = RoundEV(h17=self.h17)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        ev_cache = {}

        # 1) posbírej pool stavů (obs, ev) JEDNOU
        pool_obs = np.zeros((pool_size, BET_OBS_DIM), np.float32)
        pool_ev = np.zeros(pool_size, np.float32)
        for j in range(pool_size):
            env.reset()
            c13 = env.shoe.counts.astype(float).copy()
            pool_obs[j] = env.bet_observation().astype(np.float32)
            dealt = FULL_13 - c13
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

        # 2) trénuj Q-síť na poolu (rychlé, jen tensorové operace)
        bets = torch.arange(1, MAX_BET + 1, device=self.device).float()
        pool_obs_t = torch.as_tensor(pool_obs, device=self.device)
        pool_ev_t = torch.as_tensor(pool_ev, device=self.device)
        # Kelly/mean-variance cíl: hodnota sázky = ev*bet - risk*bet^2.
        # Rozptyl kola roste s bet^2, takže optimální sázka je ÚMĚRNÁ výhodě
        # (ne skoková min/max) — přesně jako reálné variabilní sázení.
        risk = 0.0009
        for s in range(steps):
            idx = torch.as_tensor(rng.integers(0, pool_size, size=batch),
                                  device=self.device)
            ot = pool_obs_t[idx]
            evt = pool_ev_t[idx]
            q = self.net(ot)
            target = evt[:, None] * bets[None, :] - risk * (bets[None, :] ** 2)
            loss = F.smooth_l1_loss(q, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if s % 1500 == 0 or s == steps - 1:
                print(f"  [DQN] {s}/{steps}  loss={loss.item():.5f}")

    def bet(self, bet_obs, env):
        with torch.no_grad():
            ot = torch.as_tensor(bet_obs.astype(np.float32)[None],
                                 device=self.device)
            q = self.net(ot)[0]
        return int(torch.argmax(q).item()) + 1
