# March Machine Learning Mania 2026 - Aprendizajes

**Competencia**: Predecir probabilidades de victoria para todos los matchups posibles del torneo NCAA 2026
**Metrica**: Log Loss
**Premio**: $50,000
**Deadline**: 19 de Marzo 2026
**Resultado final**: [PENDIENTE]

---

## Resultado

- **Public LB Score**: [PENDIENTE]
- **Private LB Score**: [PENDIENTE]
- **Posicion**: [PENDIENTE]
- **Best OOF CV Score**: [PENDIENTE]

---

## Que funciono

[Se completara al finalizar la competencia]

## Que NO funciono

[Se completara al finalizar la competencia]

## Feature Importance

[Se completara con los resultados de SHAP]

## Insights Estrategicos

- Log Loss castiga predicciones extremas erroneas -> siempre hacer clip [0.05, 0.95]
- Seed difference es el feature mas predictivo por lejos
- Women's tournament es mas predecible que Men's
- Temporal CV es esencial para evitar data leakage en este tipo de competencia
- Multi-seed averaging reduce varianza (comprobado en S6E2)

## Ideas para futuro

- [ ] Optuna tuning de hiperparametros
- [ ] External data (KenPom, Barttorvik)
- [ ] Opponent-adjusted stats
- [ ] Conference tournament results como features adicionales
- [ ] Elo parameter tuning (K, home advantage, carryover)
