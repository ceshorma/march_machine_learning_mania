# Experiment Backlog — 2026-03-16 (Round 29) — ALL HYPOTHESES TESTED

> **89 experimentos, 6 mejoras.** Best: **0.15534** (exp_056).
> Deadline March 19 — 3 días restantes. **Cron cancelled — exploration complete.**

## ALL 9 HYPOTHESES TESTED

| # | Hypothesis | Experiments | Best Result | Verdict |
|---|-----------|-------------|-------------|---------|
| 1 | Fatiga/Viaje | exp_079-081 | 0.15572 | Sparse coverage |
| 2 | Altura/Tamaño | exp_082-084 | 0.15621 | M-only, redundant with Elo |
| 3 | Estilo de juego | exp_079_ss | 0.15651 | Tournament-only coverage |
| 4 | Momentum mediático | exp_080_ap | 0.15651 | AP too sparse |
| 5 | Profundidad de banca | exp_083 | 0.15625 | Redundant with WinPct |
| 6 | Distribución de anotación | exp_085-086 | 0.15683 | Redundant with eFG/FTRate |
| 7 | Descanso pre-torneo | exp_079-080_rest | 0.15572 | RestDays ranked 8th, net negative |
| 8 | Overtime | exp_081_ot | 0.15696 | OT is random noise |
| 9 | Coach en torneo | N/A | N/A | Data is career-aggregate, not per-season |

## exp_056 variance analysis (3 runs)

| Run | CB Brier | Ensemble | Delta from mean |
|-----|----------|----------|-----------------|
| exp_056 (original) | 0.15824 | **0.15534** | -0.0009 (lucky) |
| exp_063 (rerun 1) | 0.15841 | 0.15613 | +0.0000 |
| exp_087 (rerun 2) | 0.15939 | 0.15629 | +0.0002 |
| **Mean** | **0.15868** | **0.15592** | — |

exp_056's 0.15534 is ~1.5σ below the mean. It's a good but somewhat lucky result.

## Final status
- **89 experiments across 29 rounds**
- **Best: 0.15534** (exp_056, submitted to Kaggle Stage 2)
- **All domain hypotheses exhausted** — no remaining ideas with meaningful probability
