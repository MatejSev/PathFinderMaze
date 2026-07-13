"""
train.py — natrénuje RL agenta (sázka + insurance + hraní).

Postup:
  1) WARM-START: supervizovaně naučí hlavy rozumné chování (sázka podle
     countu, basic strategy, insurance při vysokém countu). Vyhne se
     kolapsu na minimální sázku.
  2) PPO: doladí z reálné hry. Reward = zisk kola v jednotkách.
  3) KEEP-BEST BEZ ŠUMU: nejlepší model se vybírá podle PŘESNÉHO EV/jednotku
     (core.evaluate), ne podle zašuměného průměru kol. Tím se NEuloží model
     se šťastně vysokými sázkami — přesně chyba, co dělala model horší.

Spuštění:
    python train.py                      # rozumné defaulty
    python train.py --quick              # rychlý běh
    python train.py --iters 1200         # delší trénink

Výstup: models/rl_agent.pt (nejlepší podle bezšumové evaluace).
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from blackjack.core.environment import (BlackjackEnv, PHASE_BET, PHASE_INSURANCE,
                              PHASE_PLAY, N_BET_ACTIONS, MAX_BET_UNITS)
from blackjack.core.engine import basic_strategy
from blackjack.core.evaluate import evaluate_player
from blackjack.players.rl_network import MultiHeadAC, pad, OBS_DIM, MAX_A
from blackjack.players.rl_agent import RLAgentPlayer

from blackjack.config import PPO_MODEL, MODELS_DIR
MODEL_DIR = MODELS_DIR
MODEL_PATH = PPO_MODEL


# ---------------------------------------------------------------------------
# Warm-start (supervizované předtrénování)
# ---------------------------------------------------------------------------

def warm_start(net, steps=1500, batch=2048, h17=False, seed=0, device='cpu'):
    env = BlackjackEnv(h17=h17, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def spread_target(tc):
        if tc < 1:
            u = 1
        else:
            u = min(N_BET_ACTIONS, max(1, int(round(2 * (tc - 1)))))
        return u - 1

    for s in range(steps):
        obs_b, phase_b, act_b = [], [], []
        while len(obs_b) < batch:
            env.reset()
            done = False
            while not done:
                k = env.kind()
                if k == PHASE_BET:
                    tc = env.shoe.true_count()
                    o = pad(env.bet_observation())
                    a = spread_target(tc)
                elif k == PHASE_INSURANCE:
                    o = pad(env.insurance_observation())
                    a = 1 if env.tc_at_deal >= 3 else 0
                else:
                    h = env.player_hands[env.active]
                    m = env.play_mask()
                    o = pad(env.play_observation())
                    a = basic_strategy(h['cards'], env.dealer[0], m)
                    if not m[a]:
                        a = 1 if m[1] else 0
                obs_b.append(o)
                phase_b.append(k)
                act_b.append(a)
                _, _, done = env.step(int(a))
                if len(obs_b) >= batch:
                    break
        ot = torch.as_tensor(np.array(obs_b), device=device)
        pt = torch.as_tensor(np.array(phase_b), device=device)
        at = torch.as_tensor(np.array(act_b), device=device)
        logits, _ = net.forward(ot, pt)
        loss = F.cross_entropy(logits, at)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 500 == 0 or s == steps - 1:
            print(f"  [warm-start] {s}/{steps}  loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class Pool:
    def __init__(self, n, seed=0, h17=False):
        self.envs = [BlackjackEnv(h17=h17, seed=seed + i) for i in range(n)]
        self.n = n
        self.obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        self.phase = np.zeros(n, dtype=np.int64)
        self.mask = np.zeros((n, MAX_A), dtype=bool)
        for i in range(n):
            self.envs[i].reset()
            self._refresh(i)

    def _refresh(self, i):
        env = self.envs[i]
        k = env.kind()
        m = np.zeros(MAX_A, dtype=bool)
        if k == PHASE_BET:
            o = pad(env.bet_observation())
            m[:N_BET_ACTIONS] = True
        elif k == PHASE_INSURANCE:
            o = pad(env.insurance_observation())
            m[:2] = True
        else:
            o = pad(env.play_observation())
            m[:4] = env.play_mask()
        self.obs[i] = o
        self.phase[i] = k
        self.mask[i] = m

    def step(self, actions):
        rew = np.zeros(self.n, dtype=np.float32)
        done = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            _, r, d = self.envs[i].step(int(actions[i]))
            if d:
                rew[i] = self.envs[i].round_reward
                done[i] = 1.0
                self.envs[i].reset()
            self._refresh(i)
        return rew, done


def gae(rewards, values, dones, last_v, gamma=1.0, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + values


def train_ppo(iters=800, n_envs=1024, steps=32, h17=False, seed=0,
          warm=1500, ent0=0.02, ent1=0.008, lr0=1.5e-4, lr1=2e-5,
          eval_every=40, eval_states=120000, rew_scale=20.0):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(MODEL_DIR, exist_ok=True)
    net = MultiHeadAC().to(dev)

    print(f"Zařízení: {dev}\n[1/2] Warm-start…")
    warm_start(net, steps=warm, h17=h17, seed=seed, device=dev)
    torch.save(net.state_dict(), MODEL_PATH)

    # bezšumová evaluace po warm-startu
    best_ev = _eval_saved(h17)
    print(f"  EV/jednotku po warm-startu: {best_ev*100:+.3f}%  (uloženo)")

    opt = torch.optim.Adam(net.parameters(), lr=lr0, eps=1e-5)
    pool = Pool(n_envs, seed=seed, h17=h17)

    print(f"[2/2] PPO ({iters} iterací)…")
    t0 = time.time()
    for it in range(1, iters + 1):
        frac = it / iters
        cos = 0.5 * (1 + np.cos(np.pi * frac))
        ent_coef = ent1 + (ent0 - ent1) * cos
        for g in opt.param_groups:
            g['lr'] = lr1 + (lr0 - lr1) * cos

        Bo, Bp, Bm, Ba, Blp, Bv, Br, Bd = [], [], [], [], [], [], [], []
        for _ in range(steps):
            ot = torch.as_tensor(pool.obs, device=dev)
            pt = torch.as_tensor(pool.phase, device=dev)
            mt = torch.as_tensor(pool.mask, device=dev)
            with torch.no_grad():
                a, lp, v = net.act(ot, pt, mt)
            acts = a.cpu().numpy()
            Bo.append(pool.obs.copy()); Bp.append(pool.phase.copy())
            Bm.append(pool.mask.copy()); Ba.append(acts.copy())
            Blp.append(lp.cpu().numpy()); Bv.append(v.cpu().numpy())
            r, d = pool.step(acts)
            Br.append(r / rew_scale); Bd.append(d)

        with torch.no_grad():
            ot = torch.as_tensor(pool.obs, device=dev)
            pt = torch.as_tensor(pool.phase, device=dev)
            mt = torch.as_tensor(pool.mask, device=dev)
            _, _, last_v = net.act(ot, pt, mt)
        last_v = last_v.cpu().numpy()

        obs = np.array(Bo); phase = np.array(Bp); mask = np.array(Bm)
        act = np.array(Ba); lp = np.array(Blp); val = np.array(Bv)
        rew = np.array(Br); done = np.array(Bd)
        advs = np.zeros_like(rew); rets = np.zeros_like(rew)
        for e in range(n_envs):
            a_e, r_e = gae(rew[:, e], val[:, e], done[:, e], last_v[e])
            advs[:, e] = a_e; rets[:, e] = r_e

        f = lambda x: x.reshape(-1, *x.shape[2:])
        _ppo_update(net, opt, f(obs), f(phase).astype(np.int64), f(mask),
                    f(act).astype(np.int64), f(lp), advs.reshape(-1),
                    rets.reshape(-1), dev, ent_coef)

        if it % eval_every == 0 or it == iters:
            torch.save(net.state_dict(), os.path.join(MODEL_DIR, "_tmp.pt"))
            ev = _eval_saved(h17, path=os.path.join(MODEL_DIR, "_tmp.pt"),
                             n_states=eval_states)
            dt = time.time() - t0
            tag = ""
            if ev > best_ev:
                best_ev = ev
                torch.save(net.state_dict(), MODEL_PATH)
                tag = " *ULOŽEN (nejlepší podle přesného EV)*"
            print(f"[{it:4d}/{iters}] EV/jednotku={ev*100:+.3f}%  "
                  f"({dt:.0f}s){tag}")

    tmp = os.path.join(MODEL_DIR, "_tmp.pt")
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"\nHotovo. Nejlepší model: {MODEL_PATH}")
    print(f"EV/jednotku nejlepšího modelu: {best_ev*100:+.3f}%")


def _eval_saved(h17, path=MODEL_PATH, n_states=120000):
    """Načte uložený model jako hráče a bezšumově změří EV/jednotku."""
    player = RLAgentPlayer(path, device='cpu')
    res = evaluate_player(player, n_states=n_states, h17=h17)
    return res['ev_unit']


def _ppo_update(net, opt, obs, phase, mask, actions, logprobs, adv, ret,
                dev, ent_coef, clip=0.2, epochs=4, mb=8192, vf=0.5, mgn=0.5):
    obs = torch.as_tensor(obs, device=dev)
    phase = torch.as_tensor(phase, device=dev)
    mask = torch.as_tensor(mask, device=dev)
    actions = torch.as_tensor(actions, device=dev)
    old_lp = torch.as_tensor(logprobs, device=dev)
    adv = torch.as_tensor(adv, device=dev)
    ret = torch.as_tensor(ret, device=dev)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    N = obs.shape[0]
    idx = np.arange(N)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, N, mb):
            b = idx[s:s + mb]
            new_lp, ent, val = net.evaluate(obs[b], phase[b], actions[b], mask[b])
            ratio = torch.exp(new_lp - old_lp[b])
            s1 = ratio * adv[b]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(val, ret[b])
            loss = pi_loss + vf * v_loss - ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), mgn)
            opt.step()
