# March Machine Learning Mania 2026

Repositorio de experimentación para la competencia **March Machine Learning Mania 2026** de Kaggle.

El objetivo del proyecto fue participar en la competencia construyendo un sistema de **experimentos automatizados asistidos por IA**, donde cada iteración pudiera:

- proponer una hipótesis de basketball,
- ejecutar un experimento reproducible,
- registrar métricas y hallazgos,
- comparar contra el mejor resultado histórico,
- y decidir automáticamente cuál era el siguiente camino a explorar.

## Idea principal

En lugar de hacer pruebas manuales aisladas, este repositorio está organizado para que un agente de IA pueda trabajar como un **loop continuo de mejora**:

1. lee el estado actual del proyecto,
2. revisa backlog e hipótesis pendientes,
3. crea un nuevo experimento con un solo cambio controlado,
4. corre el pipeline completo,
5. guarda resultados, logs y submissions,
6. actualiza el registro global,
7. y prioriza la siguiente prueba.

La filosofía fue **hypothesis-first**: antes de tunear modelos, probar ideas de dominio sobre March Madness (seed history, neutral court dependency, consistency, travel, shooting profile, etc.).

## Nuevo: esqueleto genérico de AutoML

Este repositorio ahora también incluye un **framework reutilizable** para convertir este lab de March Madness en la base de otros proyectos de experimentación iterativa.

### Dónde está

- `automl_framework/` → core genérico del framework
- `examples/march_madness/` → adapter de ejemplo que conecta el framework con este repo
- `automl_state/march_madness/` → estado inicial del framework (project, backlog, registry)
- `templates/` → plantilla mínima para crear nuevos proyectos/datasets

### Qué resuelve

- separación entre lógica genérica y lógica de dominio
- registries para features, modelos y transformaciones
- entidad formal de hipótesis
- proposals con lineage parent-child
- registro estructurado de experimentos
- reglas simples de exploración para promover paths ganadores y cortar paths sin señal
- reporte automático del estado del sistema

### Uso rápido

Crear una propuesta nueva para este repositorio:

```bash
python -m automl_framework.cli march-madness-proposal --repo-root /home/runner/work/march_machine_learning_mania/march_machine_learning_mania
```

Generar el reporte del framework:

```bash
python -m automl_framework.cli march-madness-report --repo-root /home/runner/work/march_machine_learning_mania/march_machine_learning_mania
```

Inspeccionar la arquitectura operativa y la frontera exacta entre core y adapter:

```bash
python -m automl_framework.cli march-madness-architecture --repo-root /home/runner/work/march_machine_learning_mania/march_machine_learning_mania
```

Crear una plantilla genérica para otro proyecto:

```bash
python -m automl_framework.cli bootstrap-template /ruta/al/nuevo/framework_state
```

El runner histórico en `notebooks/run_experiment.py` se mantiene como implementación legacy / adapter real del caso March Madness.

### Frontera exacta entre core y adapter

- **Core (`automl_framework/`)**
  - contratos (`ProjectSpec`, `Hypothesis`, `ConfigChange`, `ExperimentProposal`, `ExperimentResult`)
  - registries de plugins
  - policy de exploración
  - generador de propuestas
  - experiment registry y reporting
  - CLI reusable

- **Adapter (`examples/march_madness/adapter.py`)**
  - define objetivo, métrica y rutas del proyecto
  - elige el baseline real
  - traduce hipótesis de basketball a cambios sobre la config legacy
  - registra features/modelos/transforms disponibles para este dominio

- **March Madness legacy sigue fuera del core**
  - `notebooks/run_experiment.py`
  - `experiments/registry.json`
  - datos NCAA/Kaggle y features específicas

### Estructura operativa

```text
.
├── automl_framework/            # Core reusable
├── examples/march_madness/      # Adapter del dominio March Madness
├── automl_state/march_madness/  # Estado del framework incubado en este repo
├── templates/                   # Plantillas para abrir proyectos nuevos
└── notebooks/run_experiment.py  # Runner legacy real del caso NCAA
```

La idea es que **este repo siga siendo la incubadora + example project**, mientras el framework reusable madura dentro de `automl_framework/`. Cuando esta frontera ya no cambie mucho, entonces sí conviene extraer un repo nuevo.

## Qué contiene

- **Pipeline modular de experimentos** en `notebooks/run_experiment.py`
- **Registro central** de resultados en `experiments/registry.json`
- **Backlog de hipótesis** en `experiments/backlog.md`
- **Reportes diarios** en `experiments/daily_reports/`
- **Experimentos versionados** en `experiments/exp_*/`
- **Notebooks y scripts auxiliares** para EDA, baseline, blending y features
- **Submissions** listas para Kaggle en `submissions/`

## Resultado destacado

Según `experiments/registry.json`:

- **Best CV Brier:** `0.15534`
- **Best experiment:** `exp_056_gender_hc_consist`
- **Best Kaggle LB:** `0.0503`
- **Total experiments:** `89`

El experimento ganador combinó:

- Elo ratings
- record / efficiency features
- KenPom + Barttorvik
- seed matchup history
- features por género
- neutral court dependency
- consistency features
- ensemble hill climbing con CatBoost + Logistic Regression

## Estructura del repositorio

```text
.
├── data/                     # Datos base de Kaggle y artefactos generados
├── experiments/              # Experimentos, registry, backlog y reportes
├── notebooks/                # Runner principal, notebooks y scripts auxiliares
│   └── utils/                # Feature engineering y carga de datos externos
├── results/                  # Resultados auxiliares
├── submissions/              # Archivos de submission para Kaggle
├── CLAUDE.md                 # Marco operativo del agente / reglas de experimentación
└── APRENDIZAJES.md           # Resumen manual de aprendizajes del proyecto
```

## Cómo correr un experimento

El runner principal se ejecuta así:

```bash
python notebooks/run_experiment.py experiments/exp_NNN/config.json
```

Cada experimento guarda típicamente:

- `config.json`
- `log.txt`
- `results.json`

Y además puede actualizar:

- `experiments/registry.json`
- archivos de `submissions/`

## Datos utilizados

### Datos base
El repositorio usa los archivos oficiales de Kaggle dentro de `data/`.

### Datos externos
También se integraron fuentes externas para enriquecer features, por ejemplo:

- KenPom
- Barttorvik
- multisource files
- coach / poll / shooting / resume features

**Importante:** el runner actual espera datos externos en una ruta local definida en código (`EXT_DIR` en `notebooks/run_experiment.py`). El valor actual es `C:/Users/Admin/Desktop/march_data_temp/`. Si quieres reproducir los experimentos en otra máquina, debes cambiar esa constante por la ruta donde tengas tus carpetas `kenpom/`, `barttorvik/` y `multisource/`.

## Dependencias

No hay un archivo de entorno cerrado en el repositorio, pero el pipeline usa principalmente:

- Python 3
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- optuna
- scipy
- jupyter

Si quieres subir resultados a Kaggle, debes configurar tu propio acceso a la API de Kaggle en tu entorno local.

## Flujo de trabajo con IA

Este proyecto fue pensado para que la IA no solo escriba código, sino que también gestione el proceso experimental:

- inspecciona resultados previos,
- detecta caminos agotados,
- prioriza hipótesis nuevas,
- evita repetir pruebas sin señal,
- documenta éxitos y fracasos,
- y mantiene trazabilidad completa de cada iteración.

En otras palabras: **no es solo un modelo para Kaggle; es un framework para mejorar el modelo continuamente**.

## Archivos clave para empezar

Si quieres entender rápido el proyecto, lee en este orden:

1. `CLAUDE.md`
2. `experiments/registry.json`
3. `experiments/backlog.md`
4. `notebooks/run_experiment.py`
5. `APRENDIZAJES.md`

## Estado del proyecto

El repositorio refleja una fase de exploración intensiva de features y ensembles para la competencia 2026. Más que un paquete genérico, es un **lab reproducible de experimentación competitiva con IA**.
