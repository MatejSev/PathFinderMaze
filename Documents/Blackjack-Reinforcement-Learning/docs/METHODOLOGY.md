# Metodologie

Tento dokument vysvětluje, jak repozitář měří kvalitu hráčů a proč jsou
výsledky spolehlivé.

## Prostředí a pravidla

Blackjack se 6 balíčky, penetrací 75 %, S17 (volitelně H17), blackjack 3:2,
double na první dvě karty, split, insurance. Prostředí (`core/environment.py`)
je fázový automat: sázka → (insurance) → hraní → vyhodnocení. Sázku,
insurance i tahy volí hráč.

## Jednotné rozhraní hráčů

Každý hráč (`players/base.py`) implementuje:

- `bet(bet_obs, env)` — velikost sázky 1..20,
- `insurance(env)` — vzít pojištění (výchozí ne),
- `play(cards, up, mask, env)` — jak hrát ruku (výchozí basic strategy).

**Všichni hrají basic strategy a sázejí plynule podle síly countu.** Liší se
jen KVALITOU odhadu výhody. Tím se izoluje přesně to, co chceme měřit —
kvalitu počítání / sázení — a ne rozdíly v hraní ruky.

## Plynulé (realistické) sázení

Sázka roste postupně s počtem (`bet_from_edge_strength` v base.py), jako u
skutečného hráče, který zvyšuje sázku s příznivějším balíčkem. Učící se
agenti (Q-learning, DQN, stealth) to dělají přes Kelly / mean-variance cíl
`ev·sázka − risk·sázka²`: protože rozptyl roste s druhou mocninou sázky,
optimální sázka vychází úměrná výhodě (ne skoková min/max).

## Bezšumové vyhodnocení — jádro spolehlivosti

Naivní přístup (hrát miliony náhodných kol a průměrovat výsledek) má
obrovský rozptyl: jedno kolo má směrodatnou odchylku ~1,1 jednotky, takže
rozlišit metody lišící se o setiny procenta by vyžadovalo stovky milionů
kol. Dřívější verze na tom ztroskotaly — pořadí metod bylo náhodné.

Řešení (`core/exact_ev.py` + `core/evaluate.py`): pro každý stav balíčku se
spočítá **přesná očekávaná hodnota kola** z jeho složení (rekurzivně, přes
composition-dependent pravděpodobnosti karet). To je deterministické, bez
šumu. Hráčova hodnota = průměr těchto přesných EV vážený jeho sázkami.
Rozptyl je tak minimální a rozdíly na setiny procenta jsou spolehlivé.

Přesnost enginu je ověřená: krupiérovy bust-rate sedí na známé hodnoty a EV
plochého basic-strategy hráče vychází ~ -0,5 % (očekávaný house edge).

## Proč je Perfect strop

Perfect (`players/perfect.py`) sází podle přesné okamžité výhody spočtené z
kompletního složení balíčku — tedy z veškeré informace, co v kartách je.
Žádný count systém nemůže mít víc informace, takže Perfect nelze překonat.
`compare.py` to kontroluje: kdyby něco Perfect přesáhlo nad rámec intervalu
spolehlivosti, je to chyba měření.

## Metriky

- **EV/jednotku** — EV na vsazenou jednotku. Hlavní metrika kvality; je
  normalizovaná na velikost sázky, takže nezvýhodňuje agresivnější spread.
- **EV/kolo** — EV na jedno kolo (roste i větší sázkou; jen doplňková).
- **korel.count** — korelace sázky s true countem. Vysoká = hráč sází
  mechanicky podle countu → kasino ho odhalí. Nízká = nenápadný.
- **spread** — poměr max/min sázky. Velký spread je sám o sobě nápadný.

## Stealth agent a konflikt zisku s nenápadností

Kasino odhaluje počítače karet podle korelace sázky s countem, velikosti
spreadu a náhlých skoků. Stealth agent (`players/stealth.py`) drží menší
spread (1..10) než plné počítání (1..20) a jen jemnou penalizaci korelace.
Kompromis (viz `analyze_stealth.py`):

```
spread 1-4    ztrátový, ale velmi nenápadný
spread 1-8    hranice ziskovosti (~0 %)
spread 1-10   ziskový (~+0,1 %) a stále diskrétnější než plný counter
spread 1-20   nejziskovější, ale kasino ho odhalí
```

Stealth agent cílí právě na spread 1-10: je ziskový, ale s méně nápadným
rozsahem sázek než klasické metody. Ukazuje, že nenápadnost a zisk jdou
skloubit — za cenu nižší výhody, než má plný counter s velkým spreadem.

## RL agenti

- **PPO** (`training/ppo.py`) — hlavní agent, učí se sázku i hru. Warm-start
  (supervizovaně k rozumnému chování) + PPO doladění. Nejlepší model se
  vybírá podle PŘESNÉHO EV (ne podle zašuměného průměru kol), takže se
  neuloží degenerovaný model.
- **Q-learning** — tabulkový, stav = diskretizovaný count. Rychlý,
  interpretovatelný.
- **DQN** — bere celé bet-observation, jemnější než tabulkový.

Všichni tři se učí na bezšumovém EV a dorovnají kvalitu klasického počítání;
Perfect nepřekonají.
