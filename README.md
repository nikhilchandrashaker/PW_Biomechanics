# The Physics of Pain: Pro Wrestling Biomechanics

A first-order biomechanical comparison of **23 pro wrestling moves** across four mechanisms (bumps, power moves, aerials, strikes) on one common scale, with Monte Carlo uncertainty and a weight-sensitivity test.

> All numbers are **model estimates** from stated assumptions, not measurements. This is an analysis project, not guidance for performing any move.

## What's inside

| Path | What it is |
|---|---|
| `paper/Wrestling_Biomechanics_Paper.pdf` | 8-page research paper: methods, 7 figures, discussion, limitations |
| `dashboard.html` | Interactive dashboard: filter by category, switch metric, click a move for details |
| `data/moves_biomechanics.csv` | Per-move results (force, g, energy, lumbar load, rotation, head accel, RLI, ranks) |
| `data/monte_carlo_summary.csv` | 5th/50th/95th percentiles from 5,000 draws per move |
| `data/move_parameters_assumptions.csv` | Every input parameter, editable |
| `figures/` | 8 PNG figures used in the paper |
| `src/simulate.py` | Physics model, Monte Carlo, index, figures |
| `src/build_paper.py`, `src/build_dashboard.py` | Paper and dashboard generators |

## The four models

- **Bumps** (4 moves): free-fall speed `v = sqrt(2gh)`, reduced by surface absorption and slap diversion; contact time = surface compliance x technique factor; peak force as a half-sine pulse.
- **Power moves** (6): same landing model for the receiver, plus lifter L5/S1 compression from a lever model, compared with NIOSH reference limits (3.4 kN action, 6.4 kN maximum).
- **Aerials** (7): projectile flight time, impact speed, contact force shared by both athletes (Newton's third law), required rotation rate, and landing-angle error for a 50 ms timing slip.
- **Strikes** (6): collision impulse with a **worked fraction** (how much of full force is delivered); head acceleration from an effective 6 kg head-neck mass.

**Relative Load Index (0-100)** = weighted sum of five components normalized to the max across moves: peak g (0.40), energy (0.20), lumbar load (0.15), rotation rate (0.15), head acceleration (0.10).

## Headline findings (baseline assumptions)

- Highest relative load: **450 splash**, **top-rope superplex**, **shooting star press**, **moonsault**, **Swanton bomb**. These five stay in the top 14 under 95% of random weightings.
- A **ladder fall (3.0 m)** hits at about 7.7 m/s versus 4.1 m/s for a standing back bump: roughly 1.9x the speed and 3.5x the energy.
- **Chokeslam** has the highest estimated lifter spine load (load held far from the body); all modeled lifts exceed the NIOSH maximum permissible limit, which is a reference point only.
- **Sanity check:** full-contact superkick/lariat head acceleration lands near 65-70 g, the same magnitude as published Olympic boxer punches (~71 g, Walilko et al. 2005). Working the strike cuts this to roughly 13-21 g.
- Strike rankings are the least stable to weights: heavy head-acceleration weighting can lift the lariat to rank 1.

## Assumptions and limitations

Point-mass bodies with empirical contact times; no rebound, joint-level loads, rotational head injury, fatigue or career exposure; move geometry is assumed, not motion-captured; the index weights are a judgment call. Edit the `MOVES` list in `src/simulate.py` (mirrored in `data/move_parameters_assumptions.csv`) to test your own assumptions.

## Run it

```bash
pip install numpy pandas matplotlib reportlab pillow
python src/simulate.py          # data + figures
python src/build_paper.py       # PDF
python src/build_dashboard.py   # dashboard.html
```

## Ideas to extend

Calibrate with video kinematics, add a repeated-exposure model for career load, model rotational head acceleration, compare ring constructions, or add a named-wrestler move database.

## References

- Waters TR, Putz-Anderson V, Garg A, Fine LJ (1993). Revised NIOSH equation for the design and evaluation of manual lifting tasks. *Ergonomics* 36(7):749-776.
- Walilko TJ, Viano DC, Bir CA (2005). Biomechanics of the head for Olympic boxer punches to the face. *Br J Sports Med* 39(10):710-719.
- Winter DA (2009). *Biomechanics and Motor Control of Human Movement*, 4th ed. Wiley.
