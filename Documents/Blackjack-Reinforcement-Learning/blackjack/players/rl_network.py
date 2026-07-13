"""
players/rl_network.py — neuronová síť RL agenta (multi-head actor-critic).

Tři hlavy sdílejí páteř: bet (velikost sázky), insurance (ano/ne), play
(hit/stand/double/split), plus value head (kritik). Používá se v tréninku
(train.py) i při hře (rl_agent.py).
"""

import numpy as np
import torch
import torch.nn as nn

from blackjack.core.environment import (BET_OBS_DIM, PLAY_OBS_DIM, N_BET_ACTIONS,
                              N_INS_ACTIONS, N_PLAY_ACTIONS,
                              PHASE_BET, PHASE_INSURANCE, PHASE_PLAY)

OBS_DIM = max(BET_OBS_DIM, PLAY_OBS_DIM)
MAX_A = max(N_BET_ACTIONS, N_INS_ACTIONS, N_PLAY_ACTIONS)


def pad(o):
    if o.shape[-1] == OBS_DIM:
        return o
    out = np.zeros((*o.shape[:-1], OBS_DIM), dtype=np.float32)
    out[..., :o.shape[-1]] = o
    return out


class MultiHeadAC(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head_bet = nn.Linear(hidden, N_BET_ACTIONS)
        self.head_ins = nn.Linear(hidden, N_INS_ACTIONS)
        self.head_play = nn.Linear(hidden, N_PLAY_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        for h in (self.head_bet, self.head_ins, self.head_play):
            nn.init.orthogonal_(h.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def _logits(self, feats, phase):
        B = feats.shape[0]
        logits = torch.full((B, MAX_A), -1e9, device=feats.device)
        lb, li, lp = self.head_bet(feats), self.head_ins(feats), self.head_play(feats)
        is_b = phase == PHASE_BET
        is_i = phase == PHASE_INSURANCE
        is_p = phase == PHASE_PLAY
        logits[is_b, :N_BET_ACTIONS] = lb[is_b]
        logits[is_i, :N_INS_ACTIONS] = li[is_i]
        logits[is_p, :N_PLAY_ACTIONS] = lp[is_p]
        return logits

    def forward(self, obs, phase):
        feats = self.body(obs)
        return self._logits(feats, phase), self.v(feats).squeeze(-1)

    def act(self, obs, phase, mask=None):
        logits, value = self.forward(obs, phase)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value

    def evaluate(self, obs, phase, actions, mask=None):
        logits, value = self.forward(obs, phase)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    @torch.no_grad()
    def greedy(self, obs, phase, mask=None):
        logits, _ = self.forward(obs, phase)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return torch.argmax(logits, dim=-1)
