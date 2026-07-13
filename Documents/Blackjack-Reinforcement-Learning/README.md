# Blackjack RL — advantage play & počítání karet

RL agent, který se učí kompletní blackjack (sázka, insurance,
hit/stand/double/split), a jeho **férové, bezšumové** porovnání s klasickými
počítacími metodami (Hi-Lo, KO, Hi-Opt II), teoretickým stropem (Perfect) a
dvěma dalšími učícími se agenty (Q-learning, DQN). Navíc **stealth agent**,
který vydělává, ale snaží se nebýt v kasinu odhalen jako počítač karet.

## Struktura repozitáře

```
blackjack_rl/
├── README.md
├── requirements.txt
├── blackjack/                    # importovatelný balíček
│   ├── config.py                 # cesty k modelům, pravidla hry
│   ├── core/                     # herní jádro
│   │   ├── engine.py             #   karty, box (shoe), basic strategy
│   │   ├── environment.py        #   prostředí (sázka → insurance → hra)
│   │   ├── exact_ev.py           #   PŘESNÉ EV složení balíčku (bez šumu)
│   │   └── evaluate.py           #   bezšumové vyhodnocení hráče
│   ├── players/                  # hráči (jednotné rozhraní, viz base.py)
│   │   ├── base.py               #   rozhraní + sdílené plynulé sázení
│   │   ├── counting.py           #   Hi-Lo, KO, Hi-Opt II
│   │   ├── perfect.py            #   Perfect (EoR) — teoretický strop
│   │   ├── rl_network.py         #   síť PPO agenta (multi-head)
│   │   ├── rl_agent.py           #   PPO agent jako hráč
│   │   ├── qlearning.py          #   tabulkový Q-learning
│   │   ├── dqn.py                #   Deep Q-Network
│   │   └── stealth.py            #   nenápadný agent
│   └── training/
│       └── ppo.py                # trénink PPO agenta (warm-start + PPO)
├── scripts/                      # spustitelné příkazy
│   ├── train_ppo.py              #   natrénuj PPO agenta
│   ├── train_qlearning.py        #   natrénuj Q-learning
│   ├── train_dqn.py              #   natrénuj DQN
│   ├── train_stealth.py          #   natrénuj stealth agenta
│   ├── compare.py                #   porovnej všechny hráče
│   ├── analyze_stealth.py        #   kompromis zisk vs. nenápadnost
│   └── play.py                   #   zahraj si (konzole, 1–7 hráčů)
├── models/                       # natrénované modely (.pt / .npy)
└── docs/
    └── METHODOLOGY.md            # jak a proč to funguje
```

## Instalace

```bash
pip install -r requirements.txt
```

Balíček není nutné instalovat — skripty si samy přidají kořen repozitáře
do cesty (viz scripts/_bootstrap.py).

## Rychlý start

```bash
# 1) porovnat všechny hráče (Q-learning/DQN/Stealth se natrénují při 1. běhu)
python scripts/compare.py --all --ppo --states 200000

# 2) natrénovat hlavního PPO agenta (GPU doporučeno)
python scripts/train_ppo.py --iters 800

# 3) zahrát si
python scripts/play.py
```

## Trénování jednotlivých agentů

```bash
python scripts/train_ppo.py          --iters 800     # -> models/ppo_agent.pt
python scripts/train_qlearning.py    --states 400000 # -> models/qlearning.npy
python scripts/train_dqn.py          --steps 6000    # -> models/dqn.pt
python scripts/train_stealth.py      --stealth 0.5   # -> models/stealth_50.pt
```

Každý skript natrénuje model od začátku, uloží do `models/` a vypíše
naučenou sázecí politiku. Přepíše existující model (slouží i k přetrénování).

## Porovnání a analýza

```bash
python scripts/compare.py                       # počítací metody + Perfect
python scripts/compare.py --ppo                 # + PPO agent
python scripts/compare.py --all --ppo           # + Q-learning, DQN, Stealth
python scripts/analyze_stealth.py               # křivka zisk vs. nenápadnost
```

## Typické výsledky (EV na vsazenou jednotku)

```
Perfect (EoR)   ~ +0.33 %      teoretický strop, nikdo ho nepřekoná
Hi-Opt II       ~ +0.17 %      silný count
Hi-Lo           ~ +0.14 %
Q-learning      ~ +0.16 %      naučené počítání, dorovná Hi-Lo
DQN             ~ +0.13 %
KO              ~ -0.10 %      nevyvážený count, nejslabší
Stealth (1-10)  ~ +0.11 %      ziskový, ale s menším spreadem než plný counter
PPO agent       roste dotrénováním k počítacím metodám
```

Podrobné vysvětlení metodologie (proč je měření bez šumu, proč je Perfect
strop, proč je stealth u nuly) je v `docs/METHODOLOGY.md`.

## Poctivé shrnutí

- **Perfect je strop daný informací v kartách. RL agent ho dorovná zdola,
  nepřekročí.** Kdyby ho něco v tabulce překonalo, je to chyba měření
  (compare.py to sám kontroluje).
- **Zisk pochází z variabilního sázení, ne z chytřejšího hraní.** Stealth
  agent proto používá menší spread (1–10) než plné počítání (1–20): je
  ziskový (~+0,11 %), ale s výrazně méně nápadným rozsahem sázek. Ukazuje,
  že nenápadnost a zisk jdou skloubit — za cenu nižší výhody než plný counter.
- **V reálném kasinu tato výhoda neplatí** (kontinuální míchačky, mělká
  penetrace, sledování hráčů). Je to korektní ukázka, PROČ advantage play
  funguje a KDE jsou jeho hranice — ne nástroj na výdělek.
