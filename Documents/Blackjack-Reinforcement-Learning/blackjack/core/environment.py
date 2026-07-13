"""
core/environment.py — herní prostředí (jeden hráč vs. krupiér).

Fázový automat: PHASE_BET -> (PHASE_INSURANCE) -> PHASE_PLAY -> konec kola.
Prostředí je JEDINÝ zdroj pravdy pro "bet observation" — vektor featur,
který dostávají VŠICHNI hráči (počítací metody i RL agent) při rozhodnutí
o sázce. Tím je vyloučeno, že by některý hráč dostal jiný/rozbitý vstup.

Sázku i insurance i hru volí hráč (viz players/). Reward = zisk kola
v jednotkách (škálovaný zvolenou sázkou).
"""

import numpy as np
from blackjack.core.engine import (RANK_VALUE, HILO_TAG, ACE, hand_value, Shoe,
                         A_STAND, A_HIT, A_DOUBLE, A_SPLIT)

PHASE_BET, PHASE_INSURANCE, PHASE_PLAY = 0, 1, 2

MAX_BET_UNITS = 20          # hráč volí 1..MAX jednotek
N_BET_ACTIONS = MAX_BET_UNITS
N_PLAY_ACTIONS = 4
N_INS_ACTIONS = 2

# bet observation: 6 featur (count-informace, dostupné PŘED rozdáním)
BET_OBS_DIM = 6
# play observation: 16 featur (stav ruky + count + sázka)
PLAY_OBS_DIM = 16


class BlackjackEnv:
    def __init__(self, num_decks=6, penetration=0.75, h17=False,
                 bj_payout=1.5, seed=None):
        self.rng = np.random.default_rng(seed)
        self.shoe = Shoe(num_decks, penetration, self.rng)
        self.h17 = h17
        self.bj_payout = bj_payout
        self._new_round()

    def _new_round(self):
        self.phase = PHASE_BET
        self.player_hands = []
        self.active = 0
        self.dealer = []
        self.tc_at_deal = 0.0
        self.base_bet = 1.0
        self.insurance = 0.0
        self.round_reward = 0.0

    def kind(self):
        return self.phase

    # -- pozorování -------------------------------------------------------

    def bet_observation(self):
        """Featury dostupné PŘED rozdáním (count + hloubka + hustoty).
        JEDINÝ zdroj pravdy pro sázecí rozhodnutí všech hráčů."""
        tc = self.shoe.true_count()
        decks_left = self.shoe.decks_left()
        rem = self.shoe.counts
        remaining = rem.sum()
        ace_den = (rem[12] / remaining - 1/13) * 13 if remaining else 0.0
        tens = rem[8:12].sum()
        ten_den = (tens / remaining - 4/13) * (13/4) if remaining else 0.0
        return np.array([
            tc / 10.0,
            np.tanh(tc / 5.0),
            decks_left / self.shoe.num_decks,
            1.0 if tc >= 2 else 0.0,
            ace_den,
            ten_den,
        ], dtype=np.float32)

    def insurance_observation(self):
        tc = self.tc_at_deal
        return np.array([
            tc / 10.0, np.tanh(tc / 5.0),
            1.0 if tc >= 3 else 0.0, self.base_bet / MAX_BET_UNITS,
        ], dtype=np.float32)

    def play_observation(self):
        if self.active >= len(self.player_hands):
            return np.zeros(PLAY_OBS_DIM, dtype=np.float32)
        h = self.player_hands[self.active]
        cards = h['cards']
        val, soft = hand_value(cards)
        up = int(RANK_VALUE[self.dealer[0]])
        can_double = len(cards) == 2 and not h['is_split_ace']
        can_split = (len(cards) == 2
                     and RANK_VALUE[cards[0]] == RANK_VALUE[cards[1]]
                     and len(self.player_hands) < 4)
        is_aa = can_split and cards[0] == ACE
        pair_val = int(RANK_VALUE[cards[0]]) if can_split else 0
        o = np.zeros(PLAY_OBS_DIM, dtype=np.float32)
        o[0] = (val - 16) / 5.0
        o[1] = 1.0 if soft else 0.0
        o[2] = (val - 16) / 5.0 if not soft else 0.0
        o[3] = (val - 18) / 3.0 if soft else 0.0
        o[4] = (up - 6.5) / 4.5
        o[5] = 1.0 if up >= 7 else 0.0
        o[6] = 1.0 if up == 11 else 0.0
        o[7] = 1.0 if can_double else 0.0
        o[8] = 1.0 if can_split else 0.0
        o[9] = 1.0 if is_aa else 0.0
        o[10] = (pair_val - 6) / 5.0
        o[11] = self.tc_at_deal / 10.0
        o[12] = len(cards) / 8.0
        o[13] = 1.0 if h['from_split'] else 0.0
        o[14] = 1.0 if val >= 21 else 0.0
        o[15] = self.base_bet / MAX_BET_UNITS
        return o

    def play_mask(self):
        h = self.player_hands[self.active]
        cards = h['cards']
        m = np.zeros(N_PLAY_ACTIONS, dtype=bool)
        m[A_STAND] = True
        m[A_HIT] = not h['is_split_ace']
        m[A_DOUBLE] = len(cards) == 2 and not h['is_split_ace']
        m[A_SPLIT] = (len(cards) == 2
                      and RANK_VALUE[cards[0]] == RANK_VALUE[cards[1]]
                      and len(self.player_hands) < 4)
        if h['is_split_ace']:
            m[:] = False
            m[A_STAND] = True
        return m

    # -- průběh kola ------------------------------------------------------

    def reset(self):
        if self.shoe.needs_shuffle():
            self.shoe.shuffle()
        self._new_round()
        return self.bet_observation()

    def _deal(self):
        self.tc_at_deal = self.shoe.true_count()
        p = [self.shoe.draw(), self.shoe.draw()]
        self.dealer = [self.shoe.draw(), self.shoe.draw()]
        self.player_hands = [{
            'cards': p, 'bet': self.base_bet, 'done': False,
            'is_split_ace': False, 'from_split': False,
        }]
        self.active = 0

    def _naturals(self):
        up = int(RANK_VALUE[self.dealer[0]])
        dval, _ = hand_value(self.dealer)
        pval, _ = hand_value(self.player_hands[0]['cards'])
        dealer_bj = (up in (10, 11) and dval == 21)
        player_bj = (pval == 21)
        b = self.base_bet
        if dealer_bj:
            self.round_reward += 0.0 if player_bj else -b
            return True
        if player_bj:
            self.round_reward += self.bj_payout * b
            return True
        return False

    def step(self, action):
        if self.phase == PHASE_BET:
            self.base_bet = float(int(action) + 1)
            self._deal()
            up = int(RANK_VALUE[self.dealer[0]])
            if up == 11:
                self.phase = PHASE_INSURANCE
                return self.insurance_observation(), 0.0, False
            if self._naturals():
                return self.play_observation(), self.round_reward, True
            self.phase = PHASE_PLAY
            return self.play_observation(), 0.0, False

        if self.phase == PHASE_INSURANCE:
            dval, _ = hand_value(self.dealer)
            dealer_bj = (dval == 21)
            if int(action) == 1:
                self.insurance = 0.5 * self.base_bet
                self.round_reward += (2.0 * self.insurance if dealer_bj
                                      else -self.insurance)
            if self._naturals():
                return self.play_observation(), self.round_reward, True
            self.phase = PHASE_PLAY
            return self.play_observation(), 0.0, False

        # PHASE_PLAY
        h = self.player_hands[self.active]
        cards = h['cards']
        if action == A_STAND:
            h['done'] = True
            return self._advance()
        if action == A_HIT:
            cards.append(self.shoe.draw())
            if hand_value(cards)[0] >= 21:
                h['done'] = True
                return self._advance()
            return self.play_observation(), 0.0, False
        if action == A_DOUBLE:
            h['bet'] *= 2.0
            cards.append(self.shoe.draw())
            h['done'] = True
            return self._advance()
        if action == A_SPLIT:
            c2 = cards.pop()
            is_ace = (cards[0] == ACE)
            new = {'cards': [c2, self.shoe.draw()], 'bet': h['bet'],
                   'done': False, 'is_split_ace': is_ace, 'from_split': True}
            cards.append(self.shoe.draw())
            h['is_split_ace'] = h['is_split_ace'] or is_ace
            h['from_split'] = True
            self.player_hands.insert(self.active + 1, new)
            if is_ace:
                h['done'] = True
                new['done'] = True
                return self._advance()
            return self.play_observation(), 0.0, False
        raise ValueError(action)

    def _advance(self):
        while (self.active < len(self.player_hands)
               and self.player_hands[self.active]['done']):
            self.active += 1
        if self.active < len(self.player_hands):
            return self.play_observation(), 0.0, False
        self._settle()
        return self.play_observation(), self.round_reward, True

    def _settle(self):
        alive = [h for h in self.player_hands
                 if hand_value(h['cards'])[0] <= 21]
        if alive:
            while True:
                dv, dsoft = hand_value(self.dealer)
                if dv < 17 or (dv == 17 and dsoft and self.h17):
                    self.dealer.append(self.shoe.draw())
                else:
                    break
        dv, _ = hand_value(self.dealer)
        dealer_bust = dv > 21
        for h in self.player_hands:
            pv, _ = hand_value(h['cards'])
            bet = h['bet']
            if pv > 21:
                self.round_reward -= bet
            elif dealer_bust or pv > dv:
                self.round_reward += bet
            elif pv < dv:
                self.round_reward -= bet
