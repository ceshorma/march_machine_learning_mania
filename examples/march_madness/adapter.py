from __future__ import annotations

import json
from pathlib import Path

from automl_framework.models import ConfigChange, Hypothesis, ProjectSpec
from automl_framework.plugins import PluginRegistry, PluginSpec
from automl_framework.runner import ProjectAdapter


class MarchMadnessAdapter(ProjectAdapter):
    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root).resolve()
        self._feature_registry = self._build_feature_registry()
        self._model_registry = self._build_model_registry()
        self._transform_registry = self._build_transform_registry()

    def project_spec(self) -> ProjectSpec:
        return ProjectSpec(
            name='march-madness-automl-skeleton',
            objective='Generate and track iterative experiment proposals for NCAA tournament prediction',
            metric='cv_brier',
            metric_mode='min',
            dataset={
                'train_assets': ['data/*.csv', 'experiments/exp_*/config.json'],
                'task_type': 'binary_classification',
                'target': 'TeamID1 wins',
            },
            splitter={'type': 'temporal_tournament_cv', 'eval_seasons': 10},
            paths={
                'repo_root': str(self.repo_root),
                'legacy_runner': str(self.repo_root / 'notebooks' / 'run_experiment.py'),
                'legacy_registry': str(self.repo_root / 'experiments' / 'registry.json'),
                'framework_state': str(self.repo_root / 'automl_state' / 'march_madness'),
            },
            metadata={
                'adapter': 'march_madness',
                'legacy_best_config': 'experiments/exp_056_gender_hc_consist/config.json',
            },
        )

    def baseline_config(self) -> tuple[str | None, dict]:
        baseline_path = self.repo_root / 'experiments' / 'exp_056_gender_hc_consist' / 'config.json'
        with open(baseline_path, 'r', encoding='utf-8') as handle:
            return 'exp_056_gender_hc_consist', json.load(handle)

    def feature_registry(self) -> PluginRegistry:
        return self._feature_registry

    def model_registry(self) -> PluginRegistry:
        return self._model_registry

    def transform_registry(self) -> PluginRegistry:
        return self._transform_registry

    def hypotheses(self) -> list[Hypothesis]:
        return [
            Hypothesis(
                id='travel_signal',
                title='Travel distance signal',
                description='Probe whether travel and timezone distance add orthogonal signal beyond Elo and seeds.',
                change_kind='feature',
                priority=95,
                path_key='feature:travel',
                success_criteria={'metric': 'cv_brier', 'direction': 'decrease'},
                tags=['domain', 'feature'],
            ),
            Hypothesis(
                id='rest_signal',
                title='Rest-days signal',
                description='Test whether conference-tournament rest improves matchup calibration.',
                change_kind='feature',
                priority=90,
                path_key='feature:rest',
                success_criteria={'metric': 'cv_brier', 'direction': 'decrease'},
                tags=['domain', 'feature'],
            ),
            Hypothesis(
                id='blend_regularizer',
                title='Linear blend regularization',
                description='Stress a lighter linear blend to see if the ensemble can improve calibration with one controlled model change.',
                change_kind='model',
                priority=70,
                path_key='model:blend',
                success_criteria={'metric': 'cv_brier', 'direction': 'decrease'},
                tags=['model', 'blend'],
            ),
            Hypothesis(
                id='clip_relaxation',
                title='Prediction clip relaxation',
                description='Use a slightly wider clip to validate whether the current calibration bounds are still optimal.',
                change_kind='transform',
                priority=50,
                path_key='transform:clip',
                success_criteria={'metric': 'cv_brier', 'direction': 'decrease'},
                tags=['calibration', 'transform'],
            ),
        ]

    def mutation_space(self) -> dict[str, list[ConfigChange]]:
        return {
            'travel_signal': [
                self.feature_registry().get('travel_distance').build_change(True, 'Enable travel distance features in the legacy config'),
            ],
            'rest_signal': [
                self.feature_registry().get('rest_days').build_change(True, 'Enable rest-day features in the legacy config'),
            ],
            'blend_regularizer': [
                self.model_registry().get('ridge').build_change(True, 'Turn on a simple ridge blender as one controlled model-family change'),
            ],
            'clip_relaxation': [
                ConfigChange(target='clip', value=[0.015, 0.985], description='Relax prediction clip by a small amount', tags=['calibration']),
            ],
        }

    def _build_feature_registry(self) -> PluginRegistry:
        registry = PluginRegistry('feature')
        registry.register(PluginSpec('travel_distance', 'feature', 'Tournament travel distance and timezone crossing features', 'features.travel_distance', False, tags=['domain']))
        registry.register(PluginSpec('rest_days', 'feature', 'Rest-days before the NCAA tournament', 'features.rest_days', False, tags=['domain']))
        registry.register(PluginSpec('seed_hist_wr', 'feature', 'Historical seed win-rate features', 'features.seed_hist_wr', True, tags=['baseline']))
        registry.register(PluginSpec('neutral_court_features', 'feature', 'Neutral-court and home-court dependency features', 'features.neutral_court_features', True, tags=['baseline']))
        registry.register(PluginSpec('consistency_features', 'feature', 'Per-team consistency statistics', 'features.consistency_features', True, tags=['baseline']))
        return registry

    def _build_model_registry(self) -> PluginRegistry:
        registry = PluginRegistry('model')
        registry.register(PluginSpec('lgb', 'model', 'LightGBM base learner', 'models.lgb', True, tags=['tree']))
        registry.register(PluginSpec('xgb', 'model', 'XGBoost base learner', 'models.xgb', True, tags=['tree']))
        registry.register(PluginSpec('cb', 'model', 'CatBoost base learner', 'models.cb', True, tags=['tree']))
        registry.register(PluginSpec('lr', 'model', 'Logistic regression blender', 'models.lr', True, tags=['linear']))
        registry.register(PluginSpec('ridge', 'model', 'Ridge blender', 'models.ridge', False, tags=['linear']))
        return registry

    def _build_transform_registry(self) -> PluginRegistry:
        registry = PluginRegistry('transform')
        registry.register(PluginSpec('auto_calibration', 'transform', 'Legacy auto calibration hook', 'calibration', 'auto', tags=['calibration']))
        registry.register(PluginSpec('clip', 'transform', 'Prediction clipping bounds', 'clip', [0.02, 0.98], tags=['calibration']))
        return registry
