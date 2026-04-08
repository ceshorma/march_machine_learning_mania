# March Machine Learning Mania 2026 — Experiment Framework

## Contexto
- Competencia: March Machine Learning Mania 2026 (NCAA basketball, Brier Score)
- Deadline: March 19, 2026
- Best CV Brier: ver `experiments/registry.json` → `best_cv_brier`
- Runner: `python notebooks/run_experiment.py experiments/exp_NNN/config.json`
- Kaggle: `KAGGLE_API_TOKEN="KGAT_11ef8ab91882f0a568ac4dc9b2d66d7f"`
- External data: `C:/Users/Admin/Desktop/march_data_temp/` (kenpom/, barttorvik/, multisource/)
- Multisource files tienen `TEAM NO` = Kaggle `TeamID` directo (sin name mapping)

---

## Hipótesis de Basketball — Feature Engineering Checklist

> **REGLA FUNDAMENTAL**: Antes de optimizar modelos (Optuna, ensemble, tuning), revisar esta tabla. Si hay hipótesis NOT TESTED con datos disponibles, probar esas primero. Un buen feature basado en una hipótesis de dominio vale más que 10 re-tunes de hiperparámetros.

### ¿Qué determina quién gana en March Madness?

| # | Hipótesis | Pregunta de basketball | Datos disponibles | Feature propuesto | Status |
|---|-----------|----------------------|-------------------|-------------------|--------|
| 1 | **Fatiga / Viaje** | ¿Equipos que viajan más al venue juegan peor? ¿El jet lag afecta? | `multisource/Tournament Locations.csv` → DISTANCE KM, DISTANCE MI, TIMEZONE CROSSING, HOME LAT/LONG, VENUE LAT/LONG | `TravelDistKM_diff`, `TimezoneCross_diff` | NOT TESTED |
| 2 | **Altura / Tamaño** | ¿Equipos más altos dominan el paint en torneo? ¿El tamaño importa más en eliminación directa? | `kenpom/Height.csv` → AvgHgt, EffHgt, C height, PF height, Experience, Bench | `AvgHeight_diff`, `EffHeight_diff`, `Experience_diff` | NOT TESTED |
| 3 | **Estilo de juego** | ¿Equipos tiradores de 3 vs equipos de paint? ¿Matchups estilísticos determinan upsets? | `multisource/Shooting Splits.csv` → Dunks FG%, Close 2s FG%, Far 2s FG%, 3PT FG% (off + def, 39 cols) | `ThreePtRate_diff`, `DunkShare_diff`, `InsideOutRatio` | NOT TESTED |
| 4 | **Momentum mediático** | ¿Equipos subiendo en polls tienen ventaja psicológica? ¿La percepción pública predice? | `multisource/AP Poll Data.csv` → YEAR, WEEK, TEAM NO, AP VOTES, AP RANK | `PollMomentum` (rank change últimas 4 semanas), `PreseasonVsFinal` | NOT TESTED |
| 5 | **Profundidad de banca** | ¿Equipos con más rotación sobreviven el desgaste del torneo? | `kenpom/Height.csv` → Bench minutes, player count, minutes distribution | `BenchDepth_diff`, `StarDependency` (% minutos del top player) | NOT TESTED |
| 6 | **Distribución de anotación** | ¿Equipos que anotan desde FT vs 3PT tienen ventaja en torneo? | `kenpom/Point Distribution.csv` → Pct from 3PT, 2PT, FT (off + def) | `PctFrom3_diff`, `PctFromFT_diff`, `ShotDiversity` | NOT TESTED |
| 7 | **Descanso pre-torneo** | ¿Más días de descanso después del conf tournament = mejor rendimiento? | `data/MConferenceTourneyGames.csv` (DayNum del último juego) + NCAA start ~DayNum 136 | `RestDaysPreTourney_diff` | NOT TESTED |
| 8 | **Overtime como señal** | ¿Equipos con muchos OT son clutch o inconsistentes? | `data/MRegularSeasonDetailedResults.csv` → NumOT | `OvertimeGames_diff`, `OvertimeWinRate_diff` | NOT TESTED |
| 9 | **Coach en torneo** | ¿Coaches con más experiencia de torneo NCAA tienen ventaja? | `multisource/Coach Results.csv` → F4%, WIN%, GAMES + `data/MTeamCoaches.csv` | `CoachTourneyGames_diff`, `CoachF4Pct_diff` | PARTIAL (career stats tested, tournament-specific no) |
| 10 | **Consistencia defensiva** | ¿Equipos con defensa consistente (bajo std dev) ganan más en torneo? | Ya implementado en `run_experiment.py` → `consistency_features` | `DefEff_Std_diff`, `OffEff_Std_diff` | TESTED ✓ (ranked 6-8th, helped on gender base) |
| 11 | **Home court dependency** | ¿Equipos que dependen mucho de jugar en casa fallan en cancha neutral? | Ya implementado → `neutral_court_features` | `HomeCourtDep_diff`, `NeutralWR_diff` | TESTED ✓ (ranked 4th, helped on gender base) |
| 12 | **Seed matchup history** | ¿Los resultados históricos de seed vs seed predicen? | Ya implementado → `seed_hist_wr`, `seed_hist_wr_by_gender` | `SeedHistWR`, `SeedHistWR_M`, `SeedHistWR_W` | TESTED ✓ (breakthrough feature) |
| 13 | **Calidad de victorias** | ¿Ganarle a equipos top importa más que ganar muchos? | `multisource/Resumes.csv` → Q1/Q2 wins, road wins, NET RPI | `Q1WinPct_diff`, `RoadWinPct_diff` | TESTED (tournament-only coverage, failed) |

### Datos externos disponibles (paths y columnas clave)

```
multisource/Tournament Locations.csv
  → YEAR, TEAM NO, ROUND, HOME CITY, HOME STATE, HOME TIMEZONE, HOME LAT, HOME LONG,
    VENUE CITY, VENUE STATE, VENUE TIMEZONE, VENUE LAT, VENUE LONG,
    DISTANCE KM, DISTANCE MI, TIMEZONE CROSSING, DIRECTION, DST FLAG

kenpom/INT _ KenPom _ Height.csv
  → Season, TeamName, Conference, AvgHgt, EffHgt, Experience,
    C_Hgt, PF_Hgt, SF_Hgt, SG_Hgt, PG_Hgt,
    Bench (52 columnas total, requires TeamName→TeamID mapping)

multisource/Shooting Splits.csv
  → YEAR, TEAM NO, Dunks_FGpct, Dunks_Share, Close2_FGpct, Close2_Share,
    Far2_FGpct, Far2_Share, 3PT_FGpct, 3PT_Share,
    Def_Dunks_FGpct, Def_Close2_FGpct, Def_3PT_FGpct (39 cols)

multisource/AP Poll Data.csv
  → YEAR, WEEK, TEAM NO, AP VOTES, AP RANK

kenpom/INT _ KenPom _ Point Distribution.csv
  → Season, TeamName, Off_FT_Pct, Off_2PT_Pct, Off_3PT_Pct,
    Def_FT_Pct, Def_2PT_Pct, Def_3PT_Pct (13 cols)

multisource/Coach Results.csv
  → YEAR, TEAM NO, COACH, F4%, WIN%, GAMES, PAKE, PASE
```

---

## Reporte Diario Matutino

Cada mañana, ANTES de correr experimentos, generar un reporte en `experiments/daily_reports/YYYY-MM-DD.md`:

### Estructura del reporte:

```markdown
# Reporte Diario — YYYY-MM-DD

## Estado actual
- Best CV Brier: X.XXXXX (exp_NNN)
- Total experimentos: N
- Días restantes: N

## Experimentos de ayer
| Exp | Hipótesis | CV Brier | Delta vs best | Veredicto |
|-----|-----------|----------|---------------|-----------|
| ... | ...       | ...      | ...           | ...       |

## Revisión de hipótesis NOT TESTED
- ¿Cuáles hipótesis de la tabla tienen datos disponibles y no se han probado?
- ¿Cuál tiene mayor probabilidad de capturar señal nueva?
- Priorizar por: (1) hipótesis de dominio fuerte, (2) datos con buena cobertura, (3) facilidad de implementación

## Análisis de patrones
- ¿Qué paths se exploraron? ¿Cuáles agotados?
- ¿Qué señales emergieron (ensemble weights, feature importance shifts)?
- ¿El modelo está en un óptimo local o hay espacio?

## Research externo
- Buscar en web: "march machine learning mania 2026 kaggle discussion"
- Buscar: "ncaa march madness prediction brier score 2026"
- ¿Hay notebooks nuevos, tips, o ideas de otros competidores?

## Plan del día
- Top 3 experimentos a intentar hoy, ordenados por impacto esperado
- **Al menos 1 debe ser una hipótesis de dominio NOT TESTED** (no solo tuning)
- Para cada uno: hipótesis clara, qué cambiar, resultado esperado
- Indicar si requiere cambios en run_experiment.py o solo config

## Actualización del backlog
- Agregar ideas nuevas descubiertas
- Repriorizar basado en patrones observados
- Mover completados, marcar callejones sin salida
```

### Cómo ejecutar:
1. Leer `experiments/registry.json` completo
2. Leer `experiments/backlog.md`
3. **Leer la tabla de hipótesis en este CLAUDE.md** — identificar NOT TESTED
4. Leer results.json de todos los experimentos desde el último reporte
5. Web search de tips recientes
6. Escribir el reporte en `experiments/daily_reports/YYYY-MM-DD.md`
7. Actualizar `experiments/backlog.md` con nuevas ideas y repriorización

---

## Loop de Experimentación (3 experimentos por sesión)

### Por cada iteración:

#### 1. Leer estado
- `experiments/registry.json` (best score, últimos experimentos)
- `experiments/backlog.md` (qué falta por probar)
- **Tabla de hipótesis en este CLAUDE.md** — ¿hay NOT TESTED?
- Reporte diario más reciente en `experiments/daily_reports/`

#### 2. Elegir siguiente experimento
- **PRIMERO**: Revisar hipótesis NOT TESTED — ¿hay alguna con datos disponibles?
- **Priorizar hipótesis de dominio** sobre optimizaciones de modelo (Optuna, ensemble, tuning)
- Seguir prioridades del reporte diario / backlog
- Regla: UNA variable a la vez (single-variable experiment)
- Si 2 experimentos consecutivos en un path fallan, cambiar de path
- Híbrido Optuna: reusar params del parent para feature changes, re-tunear para model changes

#### 3. Implementar y ejecutar
- Crear directorio: `experiments/exp_NNN_description/`
- Crear config.json basado en el mejor experimento + 1 cambio
- Si requiere cambios estructurales: modificar `run_experiment.py` primero
- Ejecutar: `python notebooks/run_experiment.py experiments/exp_NNN/config.json`

#### 4. Registrar resultados
- El runner actualiza registry.json automáticamente
- Agregar key_findings al registro
- Actualizar backlog.md (marcar completado, repriorizar)
- **Actualizar status de la hipótesis en esta tabla** (NOT TESTED → TESTED)

#### 5. Submit a Kaggle si mejoró
- **IMPORTANTE: Siempre usar el archivo STAGE 2** (`submission_stage2_*.csv`), NO stage1. Stage 1 ya cerró y da Error.
- Copiar el archivo **stage2** a `C:/Users/Admin/Desktop/` (evita brackets en path)
- `KAGGLE_API_TOKEN="KGAT_11ef8ab91882f0a568ac4dc9b2d66d7f" kaggle competitions submit -c march-machine-learning-mania-2026 -f <path> -m "exp_NNN: desc"`

---

## Reglas
- **Hypothesis-first**: Antes de optimizar modelos, agotar hipótesis de dominio con datos disponibles
- Siempre comparar contra best_cv_brier del registry
- Documentar TANTO éxitos como fracasos en key_findings
- Si un experimento falla (error), debuggear antes de continuar
- Cambios a run_experiment.py están permitidos cuando el backlog lo requiere
- Cada experimento debe tener una **hipótesis clara de basketball**, no solo "probar X"
