"""
players/rl_agent.py — RL agent jako hráč (stejné rozhraní jako ostatní).

KLÍČ: sázku i ostatní rozhodnutí bere z pozorování, které dodává PROSTŘEDÍ
(env.bet_observation / play_observation) — přesně to, na čem byl trénovaný.
Žádná ruční rekonstrukce vstupu (to byla dřívější chyba).

Ve výchozím srovnání hraje RL agent basic strategy (jako počítací metody),
aby se izolovala kvalita SÁZENÍ. Přepínačem use_own_play=True lze nechat
hrát i jeho naučenou hrací politiku.
"""

import numpy as np
import torch

from blackjack.players.base import Player
from blackjack.players.rl_network import MultiHeadAC, pad
from blackjack.core.environment import PHASE_BET, PHASE_INSURANCE


class RLAgentPlayer(Player):
    name = "RL agent"

    def __init__(self, model_path, device='cpu', use_own_play=False,
                 use_own_insurance=False):
        self.device = device
        self.net = MultiHeadAC().to(device)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()
        self.use_own_play = use_own_play
        self.use_own_insurance = use_own_insurance

    def bet(self, bet_obs, env):
        o = pad(bet_obs.astype(np.float32))
        ot = torch.as_tensor(o[None], device=self.device)
        pt = torch.as_tensor(np.array([PHASE_BET]), device=self.device)
        a = int(self.net.greedy(ot, pt)[0])
        return a + 1

    def insurance(self, env):
        if not self.use_own_insurance:
            return False
        o = pad(env.insurance_observation())
        ot = torch.as_tensor(o[None], device=self.device)
        pt = torch.as_tensor(np.array([PHASE_INSURANCE]), device=self.device)
        return int(self.net.greedy(ot, pt)[0]) == 1

    def play(self, cards, dealer_up_rank, legal_mask, env):
        if not self.use_own_play:
            from blackjack.core.engine import basic_strategy
            return basic_strategy(cards, dealer_up_rank, legal_mask)
        from blackjack.core.environment import PHASE_PLAY
        o = pad(env.play_observation())
        ot = torch.as_tensor(o[None], device=self.device)
        pt = torch.as_tensor(np.array([PHASE_PLAY]), device=self.device)
        mt = torch.as_tensor(legal_mask[None], device=self.device)
        a = int(self.net.greedy(ot, pt, mt)[0])
        if not legal_mask[a]:
            a = 1 if legal_mask[1] else 0
        return a
